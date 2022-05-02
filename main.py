from simpletransformers.classification import ClassificationModel
from rdkit import Chem
import argparse
def parse_args():

    parser = argparse.ArgumentParser(description='Run Chemical explosion prediction from command line')
    parser.add_argument('-s', '--smiles', default=None, type = str, 
                        help='SMILES input for yield predictions')
    parser.add_argument('-n', '--name', default='test_mol', help='The name of the molecule')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if bool(args.smiles) == True:
        if bool(Chem.MolFromSmiles(args.smiles)) == True:
            model = ClassificationModel('roberta', 'Parsa/Chemical_explosion_classification',use_cuda=False)
            pred, _ = model.predict([args.smiles])
            if pred == 1:
                print('Explosive')
            else:
                print('Not Explosive')

        else:
            print('Invalid smiles')
    else:
        print('Empty input')
