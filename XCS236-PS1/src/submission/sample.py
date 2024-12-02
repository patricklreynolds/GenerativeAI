import torch
import torch.nn.functional as F
from tqdm import trange

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def temperature_scale(logits, model, new_past, config, temperature, temperature_horizon):
    if temperature is None:
        return logits
    if temperature_horizon == 1:
        scaled_logits = logits / temperature
        return scaled_logits
    elif temperature_horizon == 2:
        first_token_logits = logits / temperature
        first_token_probs = F.softmax(first_token_logits, dim=-1)
        return_logits = torch.zeros_like(first_token_logits)
        for i in range(config.vocab_size):
            current_token = torch.full((logits.shape[0], 1), i, dtype=torch.long, device=logits.device)
            second_logits, _ = model(current_token, new_past)
            second_logits = second_logits[:, -1, :] / temperature
            second_probs = F.softmax(second_logits, dim=-1)
            joint_probs = first_token_probs[:, i].unsqueeze(1) * second_probs
            marginal_prob = joint_probs.sum(dim=1)
            return_logits[:, i] = marginal_prob.log()
        return return_logits


def sample(model, start_text, config, length, temperature=None, temperature_horizon=1):
    current_text = start_text
    past = None
    output = [start_text]
    with torch.no_grad():
        for _ in trange(length):
            logits, new_past = model(current_text, past=past)
            # Input parameters:
            #     current_text: the encoded text token at t-1
            #     past: the calculated hidden state of previous text or None if no previous text given
            # Return:
            #     logits: a tensor of shape (batch_size, sequence_length, size_of_vocabulary)
            #     past: the calculated hidden state of previous + current text

            current_logits = logits[:, -1, :]
            logits = top_k_logits(current_logits, k=config.top_k)
            logits = temperature_scale(logits, model, new_past, config, temperature, temperature_horizon)
            
            ### START CODE HERE ###
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            next_token = torch.multinomial(probs, num_samples=1)  # Sample the next token
            output.append(next_token)  # Append the sampled token to the output list
            current_text = next_token  # Update the current text for the next iteration
            ### END CODE HERE ###

            past = new_past  # Update past to the new past for next iteration

        output = torch.cat(output, dim=1)  # Concatenate the list of tokens into a tensor
        return output
