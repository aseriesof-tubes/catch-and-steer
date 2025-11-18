import torch, tqdm 

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