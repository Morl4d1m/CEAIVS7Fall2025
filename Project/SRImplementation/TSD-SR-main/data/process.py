import os
import sys
sys.path.append(os.getcwd())

from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision import transforms
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModelWithProjection, T5TokenizerFast
from diffusers import AutoencoderKL

# Paths to the datasets
FLICKR2K_PATH = "/path/to/your/Flickr2K/datasets"  
DIV2K_PATH = "/path/to/your/DIV2K/datasets"        
LSDIR80K_PATH = "/path/to/your/LSDIR/datasets"    
FFHQ10K_PATH = "/path/to/your/FFHQ/datasets"     

data_path = [LSDIR80K_PATH, DIV2K_PATH, FFHQ10K_PATH, FLICKR2K_PATH]
sd3_path = "/path/to/your/sd3_model"  # Path to the SD3 model
hr_dir_name = "gt" # High-resolution images directory
prompt_dir_name = "prompt_txt" # Prompt text directory

# Embedding save path
prompt_embeds_dir_name = "prompt_embeds" 
pool_prompt_embeds_dir_name = "pool_embeds"
hr_latnet_dir_name = "latent_hr"

def merge_data(data_path, hr_name=hr_dir_name):
    """merge dataset path"""
    hr_data_file_path = []
    for data_dir in data_path:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} not exist")
        
        hr_data_dir = os.path.join(data_dir, hr_name)
        
        hr_file = os.listdir(hr_data_dir)
        hr_file.sort(key=lambda x: int(x.split(".")[0]))
        
        sub_hr_data_dir = [os.path.join(hr_data_dir , file) for file in hr_file]
        
        hr_data_file_path = hr_data_file_path + sub_hr_data_dir
    
    return hr_data_file_path

# ---------------------- text embedding ---------------------- #
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

def import_text_encoder(pretrained_model_name_or_path=sd3_path, device=None): 
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_3",
    )

    # import correct text encoder classes
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2").to(device)
    text_encoder_three = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_3").to(device)
    
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    return tokenizers, text_encoders

def run_encode_prompt(data_dir):
    hr_data_file = merge_data(data_dir)
    for data_file in data_dir:
        if not os.path.exists(os.path.join(data_file, prompt_embeds_dir_name)):
            os.makedirs(os.path.join(data_file, prompt_embeds_dir_name))
        if not os.path.exists(os.path.join(data_file, pool_prompt_embeds_dir_name)):
            os.makedirs(os.path.join(data_file, pool_prompt_embeds_dir_name))
            
    tokenizers ,text_encoders = import_text_encoder(device="cuda")

    for hr_img_file in tqdm(hr_data_file, total=len(hr_data_file)):
        prompt_file = hr_img_file.replace(".png",".txt").replace(hr_dir_name, prompt_dir_name)
        prompt_path = prompt_file.replace(".txt",".pt").replace(prompt_dir_name, prompt_embeds_dir_name)
        pool_path = prompt_file.replace(".txt",".pt").replace(prompt_dir_name, pool_prompt_embeds_dir_name)
        
        with open(prompt_file, "r") as f:
            prompt = f.read()
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, device="cuda")
        torch.save(prompt_embeds.detach().cpu(), prompt_path)    
        torch.save(pooled_prompt_embeds.detach().cpu(), pool_path)    
        print("{} Done !".format(prompt_path))
        del prompt_embeds, pooled_prompt_embeds
        torch.cuda.empty_cache()
        
# ---------------------- vae encode ---------------------- #
def vae_encode(hr_img_paths, hr_latent_path=hr_latnet_dir_name, sd3_model_path=sd3_path, weight_dtype=torch.float32):
    vae = AutoencoderKL.from_pretrained(sd3_model_path, subfolder="vae").to("cuda", weight_dtype)
    trans = transforms.ToTensor()
    for hr_img_file in tqdm(hr_img_paths, total=len(hr_img_paths)):
        hr_save_path = hr_img_file.replace(".png",".pt").replace(hr_dir_name, hr_latent_path)
        hr_img = Image.open(hr_img_file).convert("RGB")
        hq = trans(hr_img).unsqueeze(0).to("cuda",dtype=weight_dtype) * 2 - 1
        hq_latent = vae.encode(hq).latent_dist.sample() * vae.config.scaling_factor
        torch.save(hq_latent.detach().cpu(), hr_save_path)
        print("{} Done !".format(hr_save_path.split("/")[-1]))
        del hq, hq_latent
        torch.cuda.empty_cache()

def run_vae_encode(data_dir):
    hr_data_file = merge_data(data_dir)
    for data_file in data_dir:
        if not os.path.exists(os.path.join(data_file, hr_latnet_dir_name)):
            os.makedirs(os.path.join(data_file, hr_latnet_dir_name))
    vae_encode(hr_data_file)

if __name__ == '__main__':
    run_vae_encode(data_path)
    run_encode_prompt(data_path)
    print("All done !")
