import argparse
import torch
import torchvision
from utils import *
import script_utils
import os

def main():

    args = create_argparser().parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        for epoch in range(200,2000,200):

            diffusion = script_utils.get_diffusion_from_args(args).to(device)
            
            # Please modify the model file path according to the actual situation.
            print("Please modify the model file path according to the actual situation.")
            model_path = f'../model/E_coli_promoter-ddpm-2024-04-22-15-58-iteration-{epoch}--kernel=3--model.pth'
            
            print(' model_path = ', model_path)

            diffusion.load_state_dict(torch.load(model_path))
            sequences = []

            for i in range(2):

                print('strat to generate sequences')
                samples = diffusion.sample(args.num_images, device)
                print('end to generate sequences')

                samples = samples.squeeze(dim=1)
                samples = samples.to('cpu').detach().numpy()

                for i in range(samples.shape[0]):

                    decoded_sequence = decode_one_hot(samples[i])
                    sequences.append(decoded_sequence)

                make_fasta_file(sequences,path=f'../sequences/E_coli_promoter-ddpm-2024-04-22-15-58-iteration-{epoch}.fasta')

    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")

def create_argparser():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1024, device=device, schedule_low=1e-4,schedule_high=0.02,out_init_conv_padding = 1)

    defaults.update(script_utils.diffusion_defaults())
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)

    script_utils.add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
