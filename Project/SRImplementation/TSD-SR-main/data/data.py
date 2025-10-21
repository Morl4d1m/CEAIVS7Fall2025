import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Paths to the datasets
FLICKR2K_PATH = "/path/to/your/Flickr2K/datasets"  
DIV2K_PATH = "/path/to/your/DIV2K/datasets"        
LSDIR80K_PATH = "/path/to/your/LSDIR/datasets"    
FFHQ10K_PATH = "/path/to/your/FFHQ/datasets"        

data_path = [LSDIR80K_PATH, DIV2K_PATH, FFHQ10K_PATH, FLICKR2K_PATH]
lr_dir_name = "lr_bicubic"
hr_dir_name = "gt"
prompt_dir_name = "prompt_txt"
prompt_embeds_dir_name = "prompt_embeds"
pool_prompt_embeds_dir_name = "pool_embeds"
hr_latnet_dir_name = "latent_hr"

class Real_ESRGAN_Dataset(Dataset):
    def __init__(self,
                root_dir_path=data_path, 
                process_size=512,
                max_sample=None,
                device="cpu"
                ):
        self.device = device
        self.max_sample = max_sample
        self.process_size=process_size
        self.lr_dir_name = lr_dir_name
        self.hr_dir_name = hr_dir_name
        self.hr_latnet_dir_name = hr_latnet_dir_name
        self.prompt_dir_name = prompt_dir_name
        self.prompt_embeds_dir_name = prompt_embeds_dir_name
        self.pool_prompt_embeds_dir_name = pool_prompt_embeds_dir_name
        self.trans = transforms.ToTensor()
        
        self.lr_img_name = []
        self.hr_img_name = []
        self.lr_latent_name = []
        self.hr_latent_name = []
        self.prompt_name = []
        self.prompt_embeds_name = []
        self.pool_prompt_embeds_name = []
        
        for root_dir in root_dir_path:
            # image data path
            lr_data_path = os.path.join(root_dir, self.lr_dir_name)
            hr_data_path = os.path.join(root_dir, self.hr_dir_name)
            latent_hr_path = os.path.join(root_dir, self.hr_latnet_dir_name)
            # prompt path
            self.prompt_path = os.path.join(root_dir, self.prompt_dir_name)
            self.prompt_embeds_path = os.path.join(root_dir, self.prompt_embeds_dir_name)
            self.pool_prompt_embeds_path = os.path.join(root_dir, self.pool_prompt_embeds_dir_name)
            
            data_file = os.listdir(lr_data_path)
            data_file.sort(key=lambda x: int(x.split(".")[0]))
            
            self.lr_img_name = self.lr_img_name + [os.path.join(lr_data_path, file) for file in data_file]
            self.hr_img_name = self.hr_img_name + [os.path.join(hr_data_path, file) for file in data_file]
            self.hr_latent_name = self.hr_latent_name + [os.path.join(latent_hr_path, file.replace(".png", ".pt")) for file in data_file]
            
            self.prompt_name = self.prompt_name + [os.path.join(self.prompt_path, file.replace(".png", ".txt")) for file in data_file]
            self.prompt_embeds_name = self.prompt_embeds_name + [os.path.join(self.prompt_embeds_path, file.replace(".png", ".pt")) for file in data_file]
            self.pool_prompt_embeds_name = self.pool_prompt_embeds_name + [os.path.join(self.pool_prompt_embeds_path, file.replace(".png", ".pt")) for file in data_file]
            
        self.img_nums = len(self.lr_img_name)
    
    def __getitem__(self, idx):
        lr_img = self.trans(Image.open(self.lr_img_name[idx]).convert("RGB")).squeeze() * 2 - 1 
        hr_img = self.trans(Image.open(self.hr_img_name[idx]).convert("RGB")).squeeze() * 2 - 1 
        latent_hr = torch.load(self.hr_latent_name[idx], map_location=self.device).squeeze() 
        with open(self.prompt_name[idx], "r") as f:
            prompt = f.read()
        prompt_embeds = torch.load(self.prompt_embeds_name[idx], map_location=self.device).squeeze()
        pooled_prompt_embeds = torch.load(self.pool_prompt_embeds_name[idx], map_location=self.device).squeeze()
            
        return {
            "lr_img": lr_img,
            "hr_img": hr_img,
            "latent_hr": latent_hr,
            "prompt_embeds_input": prompt_embeds,
            "pooled_prompt_embeds_input": pooled_prompt_embeds,
            "prompt_text": prompt,
            }

    def __len__(self):
        if self.max_sample:
            return self.max_sample
        return self.img_nums
    
if __name__ == "__main__":
    dataset = Real_ESRGAN_Dataset()
    for i in range(len(dataset)):
        print(len(dataset))
        data = dataset[i]
        print(data["lr_img"].shape)  # [3, 512, 512]
        print(data["hr_img"].shape)  # [3, 512, 512]
        print(data["latent_hr"].shape) # [16, 64, 64]
        print(data["prompt_embeds_input"].shape) # [333, 4096]
        print(data["pooled_prompt_embeds_input"].shape) # [2048]
        print(data["prompt_text"])
        break