import torch
from tqdm import tqdm
from einops import einsum, rearrange, repeat

device = 'cuda'

def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids


def find_steering_vecs(model, base_toks, target_toks, batch_size = 16): 
    '''
    We want to find the steering vector from base_toks to target_toks (we do target_toks - base_toks)
    Inputs: 
        :param model: the model to use
        :param base_toks: the base tokens [len, seq_len]
        :param target_toks: the target tokens [len, seq_len]
    Output: 
        :return steering_vecs: the steering vectors [hidden_size]
    '''
    device = model.device
    num_its = len(range(0, base_toks.shape[0], batch_size))
    steering_vecs = {}
    for i in tqdm(range(0, base_toks.shape[0], batch_size)): 
        # pass through the model 
        base_out = model(
            base_toks[i:i+batch_size].to(device), 
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        ).hidden_states # tuple of length num_layers with each element size [batch_size, seq_len, hidden_size]

        target_out = model(target_toks[i:i+batch_size].to(device),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True
        ).hidden_states
        for layer in range(len(base_out)): 
            # average over the batch_size, take last token 
            if i == 0: 
                steering_vecs[layer] = torch.mean(target_out[layer][:,-1,:].detach().cpu() - base_out[layer][:,-1,:].detach().cpu(), dim=0)/num_its # [hidden_size]
            else: 
                steering_vecs[layer] += torch.mean(target_out[layer][:,-1,:].detach().cpu() - base_out[layer][:,-1,:].detach().cpu(), dim=0)/num_its
    return steering_vecs

def do_single_steering(model, test_toks, steering_vec, scale = 1, normalise = True, layer = None, proj=True, batch_size=16): 
    '''
    Input: 
        :param model: the model to use
        :param test_toks: the test tokens [len, seq_len]
        :param steering_vec: the steering vector [hidden_size]
        :param scale: the scale to use
        :param layer: the layer to modify; if None: we modify all layers. 
        :param proj: whether to project the steering vector
    Output:
        :return output: the steered model output [len, generated_seq_len]
    '''
    # define a hook to modify the input into the layer
    if steering_vec is not None: 
        def modify_activation():
            def hook(model, input): 
                if normalise:
                    sv = steering_vec / steering_vec.norm()
                else: 
                    sv = steering_vec
                if proj:
                    sv = einsum(input[0], sv.view(-1,1), 'b l h, h s -> b l s') * sv
                input[0][:,:,:] = input[0][:,:,:] - scale * sv
            return hook
        handles = [] 
        for i in range(len(model.model.layers)):
            if layer is None: # append to each layer
                handles.append(model.model.layers[i].register_forward_pre_hook(modify_activation()))
            elif layer is not None and i == layer:
                handles.append(model.model.layers[i].register_forward_pre_hook(modify_activation()))

    gen_kwargs = dict(num_beams=4, do_sample=True, max_new_tokens=60, use_cache=False)

    # pass through the model
    outs_all = []
    for i in tqdm(range(0, test_toks.shape[0], batch_size)):
        batch = test_toks[i:i+batch_size].to(device)
        outs = model.generate(batch,**gen_kwargs) # [num_samples, seq_len]
        outs_all.append(outs)
    outs_all = torch.cat(outs_all, dim=0)
    # remove all hooks
    if steering_vec is not None: 
        for handle in handles: 
            handle.remove()
    return outs_all

def do_multi_steering(
    model,
    test_toks,
    steering_vecs_list,
    scales_list,
    normalise=True,
    layer=None,
    proj="none",      # options: False / "none", "k-dir", "1-dir"
    batch_size=16,
):
    '''
    Multi-steering: Apply multiple steering vectors simultaneously.

    Inputs:
        :param model: the model to use
        :param test_toks: the test tokens [len, seq_len]
        :param steering_vecs_list: list of steering vectors, each [hidden_size]
        :param scales_list: list of scales for each steering vector (same length)
        :param normalise: whether to normalize each steering vector before use
        :param layer: the layer to modify; if None: we modify all layers.
        :param proj: projection mode:
            - False or "none": no projection, direct additive steering
            - "k-dir": project onto each steering direction separately and sum
            - "1-dir": combine steering directions into one and project once
        :param batch_size: batch size for generation

    Output:
        :return outs_all: the steered model output [len, generated_seq_len]
    '''
    device = model.device

    # Basic checks
    if steering_vecs_list is not None and len(steering_vecs_list) > 0:
        assert len(steering_vecs_list) == len(scales_list), \
            "steering_vecs_list and scales_list must have the same length"

        # Normalize proj argument and keep backward-compat for True/False
        if proj is False or proj is None:
            proj_mode = "none"
        elif proj is True:
            proj_mode = "k-dir"   # backward-compatible default
        else:
            proj_mode = proj

        def modify_activation():
            def hook(m, inp):
                x = inp[0]  # [batch, seq_len, hidden]
                if x is None:
                    return

                # Prepare steering vectors: move to device and optionally normalize
                prepared = []
                for sv, scale in zip(steering_vecs_list, scales_list):
                    sv = sv.to(x.device)
                    if normalise:
                        norm = sv.norm()
                        if norm > 0:
                            sv = sv / norm
                    prepared.append((sv, scale))

                if proj_mode == "none":
                    # No projection: direct additive steering (broadcast over batch & seq)
                    combined = torch.zeros_like(x)
                    for sv, scale in prepared:
                        # sv: [h] → broadcast to [b, l, h]
                        combined = combined + scale * sv
                    x[:, :, :] = x[:, :, :] - combined

                elif proj_mode == "k-dir":
                    # Project onto each direction separately and sum
                    combined = torch.zeros_like(x)
                    for sv, scale in prepared:
                        # x: [b, l, h], sv: [h] → [h, 1]
                        coeff = einsum(x, sv.view(-1, 1), 'b l h, h s -> b l s')  # [b, l, 1]
                        # coeff * sv: [b, l, 1] * [h] → [b, l, h] (broadcast)
                        proj_x = coeff * sv
                        combined = combined + scale * proj_x
                    x[:, :, :] = x[:, :, :] - combined

                elif proj_mode == "1-dir":
                    # Combine directions into one effective steering direction
                    # Start from zero vector in hidden space
                    v_total = None
                    for sv, scale in prepared:
                        contrib = scale * sv
                        v_total = contrib if v_total is None else (v_total + contrib)

                    # If normalise=True, make this a unit vector so the operation is
                    # a proper projection onto a single direction
                    if normalise:
                        norm = v_total.norm()
                        if norm > 0:
                            v_dir = v_total / norm
                        else:
                            v_dir = v_total
                    else:
                        v_dir = v_total

                    # Project x onto v_dir
                    coeff = einsum(x, v_dir.view(-1, 1), 'b l h, h s -> b l s')  # [b, l, 1]
                    proj_x = coeff * v_dir  # [b, l, h]
                    x[:, :, :] = x[:, :, :] - proj_x

                else:
                    raise ValueError(f"Invalid proj value: {proj_mode}")

            return hook

        # Register hooks
        handles = []
        num_layers = len(model.model.layers)
        for i in range(num_layers):
            if layer is None:
                handles.append(
                    model.model.layers[i].register_forward_pre_hook(modify_activation())
                )
            elif i == layer:
                handles.append(
                    model.model.layers[i].register_forward_pre_hook(modify_activation())
                )
    else:
        handles = []

    gen_kwargs = dict(num_beams=4, do_sample=True, max_new_tokens=60, use_cache=False)

    # Pass through the model
    outs_all = []
    for i in tqdm(range(0, test_toks.shape[0], batch_size)):
        batch = test_toks[i:i+batch_size].to(device)
        outs = model.generate(batch, **gen_kwargs)
        outs_all.append(outs)
    outs_all = torch.cat(outs_all, dim=0)

    # Remove hooks
    for h in handles:
        h.remove()

    return outs_all
