from dataset import create_BOLD5000_dataset, identity, remove_duplicates_from_dataset
import torch
from torch.utils.data import DataLoader
import argparse

def main(subjects=['CSI1'], batch_size=8, path_BOLD_dataset='../data/BOLD5000', path_save='../data/BOLD5000/CSI1_no_duplicates.pth'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train, test = create_BOLD5000_dataset(path=path_BOLD_dataset, subjects=subjects)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=True)
    print("Loaded train and test sets")
    
    train = remove_duplicates_from_dataset(train_dl)
    test = remove_duplicates_from_dataset(test_dl)
    print("Removed duplicates")

    to_save = {'train': train, 'test': test}
    torch.save(to_save, path_save)
    print(f"Saved to {path_save}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--path', type=str, required=True, help='path to the BOLD5000 folder')
    parser.add_argument('-p2', '--save_path', type=str, required=True, help='location to save file')
    parser.add_argument('-s', '--subjects', type=str, nargs='+', help='List of subjects to use')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size to use')

    args = parser.parse_args()

    main(args.subjects, args.batch_size, args.path, args.save_path)