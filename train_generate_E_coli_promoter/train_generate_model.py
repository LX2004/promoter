import argparse
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
import script_utils
from torch.optim.lr_scheduler import StepLR
from utils import *
import pdb

class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data = data_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

def main():

    # loss_flag = 0.15
    args = create_argparser().parse_args()
    
    model_path = args.log_dir
    if not os.path.exists(model_path):
        os.makedirs(folder_path)
        print(f"Created folder: {model_path}")
    else:
        print(f"Folder already exists: {model_path}")
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))

        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        batch_size = args.batch_size
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        if args.project_name == 'E_coli':

            data = np.load('../data/promoter.npy')

            all_sample_number = data.shape[0]
            all_sample = data

        else:
            print('error!!!')

        train_size = int(0.9 *  all_sample_number)  
        encoded_sequence_train = []

        for sequence in all_sample[:train_size]:

            if len(sequence) != 50:
                print('error!!!')

            encoded_sequence = one_hot_encoding(sequence)
            encoded_sequence_train.append(encoded_sequence)
            
        encoded_sequence_test = []
        for sequence in all_sample[train_size:]:

            if len(sequence) != 50:
                print('error!!!')
            
            encoded_sequence = one_hot_encoding(sequence)
            encoded_sequence_test.append(encoded_sequence)

        train_arrary = np.array(encoded_sequence_train)
        test_arrary = np.array(encoded_sequence_test)

        train_arrary = np.expand_dims(train_arrary, axis=1)
        test_arrary = np.expand_dims(test_arrary, axis=1)

        train_dataset = CustomDataset(train_arrary)
        test_dataset = CustomDataset(test_arrary)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Early stopping variables
        best_test_loss = float('inf')
        epochs_since_improvement = 0
        
        for iteration in range(1, args.iterations + 1):

            diffusion.train()
            acc_train_loss = 0
            
            for x in train_loader:

                x = x.to(device)               
                loss = diffusion(x)

                acc_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                diffusion.update_ema()
                
            acc_train_loss /= len(train_loader)
            scheduler.step()
            
            if iteration % args.log_rate == 0:
                test_loss = 0

                with torch.no_grad():

                    diffusion.eval()
                    
                    for x in test_loader:

                        x = x.to(device)
                        loss = diffusion(x)
                        test_loss += loss.item()
                
                # samples = diffusion.sample(10, device)
                # samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate
            
                print(f'epoch = {iteration}, train loss = {acc_train_loss}')
                print(f'epoch = {iteration}, test_loss = {test_loss}')
            
            # Early stopping logic
                if test_loss < best_test_loss:
                    
                    best_test_loss = test_loss
                    epochs_since_improvement = 0  # Reset the counter when improvement occurs
                    print('Best model saved')

                    model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-kernel={1+2*args.out_init_conv_padding}--best-model.pth"
                    torch.save(diffusion.state_dict(), model_filename)
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= args.early_stopping:
                    print(f"Early stopping triggered after {iteration} iterations")
                    break

            # if test_loss < loss_flag:

            #     loss_flag = test_loss
            #     print('save best model')

            #     model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-kernel={1+2*args.out_init_conv_padding}--best-model.pth"
            #     torch.save(diffusion.state_dict(), model_filename)

            if iteration % args.checkpoint_rate == 0:

                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}--kernel={1+2*args.out_init_conv_padding}--model.pth"
                torch.save(diffusion.state_dict(), model_filename)

    except KeyboardInterrupt:

        print("Keyboard interrupt, run finished early")


def create_argparser():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")

    defaults = dict(

        learning_rate=1e-4,
        batch_size=512,
        iterations=2000,

        log_to_wandb=True,
        log_rate=1,
        checkpoint_rate=200,
        log_dir="../model",

        project_name='E_coli',
        out_init_conv_padding = 1,
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
        # Early stopping parameter (0 means no early stopping)
        early_stopping=10,
    )

    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
