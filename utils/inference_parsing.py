##########################################################################
# File Name: inference_parsing.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Thu 19 Oct 2023 09:31:15 AM CST
#########################################################################

from argparse import ArgumentParser, FileType

def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", 
        type=FileType(mode="r"), 
        default=None
    )
    parser.add_argument(
        "--protein_peptide_csv",
        type=str,
        default=None,
        help="Path to a .csv file specifying the multiple inputs as described in the README. If this is not None, it will be used instead of the --protein_description and --peptide_description parameters",
    )
    parser.add_argument(
        "--complex_name",
        type=str,
        default=None,
        help="Name that the docked complex result will be saved with",
    )
    parser.add_argument(
        "--protein_description",
        type=str,
        default=None,
        help="Either the path to a protein .pdb file or a sequence of the input protein for ESMFold",
    )
    parser.add_argument(
        "--peptide_description",
        type=str,
        default=None,
        help="Either the path to a peptide .pdb file or a sequence of the input peptide",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/default_result",
        help="Directory where the outputs will be written to",
    )
    parser.add_argument(
        "--save_visualisation",
        action="store_true",
        default=False,
        help="Save a .pdb file with all of the steps of the reverse diffusion",
    )
    parser.add_argument(
        "--N", type=int, default=None, help="Number of samples to generate"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to folder with trained score model and hyperparameters",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint to use for the score model",
    )
    parser.add_argument(
        "--scoring_function",
        type=str,
        default=None,
        help="The scoring function to use (confidence/ref2015)",
    )
    parser.add_argument(
        "--fastrelax",
        action="store_true",
        default=False,
        help="Use FastRelax to optimize generated peptide. This option is on if --scoring_function is chosed to be ref2015",
    )
    parser.add_argument(
        "--confidence_model_dir",
        type=str,
        default=None,
        help="Path to folder with trained confidence model and hyperparameters, this is used if --scoring_function is chosed to be confidence",
    )
    parser.add_argument(
        "--confidence_ckpt",
        type=str,
        default=None,
        help="Checkpoint to use for the confidence model, this is used if --scoring_function is chosed to be confidence",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="The batch size used in inference process",
    )
    parser.add_argument(
        "--no_final_step_noise",
        action="store_true",
        default=False,
        help="Use no noise in the final step of the reverse diffusion",
    )
    parser.add_argument(
        "--inference_steps", type=int, default=None, help="Number of denoising steps"
    )
    parser.add_argument(
        "--actual_steps",
        type=int,
        default=None,
        help="Number of denoising steps that are actually performed",
    )

    parser.add_argument(
        "--conformation_partial",
        type=str,
        default=None,
        help="The partial of initial type of peptide conformation. H:E:P.",
    )
    parser.add_argument(
        "--conformation_type",
        type=str,
        default="H",
        help="The initial type of peptide conformation. H: Helical conformation: φ = -57°, ψ = -47°; E: Extended conformation: φ = -139°, ψ = 135°; P: Polyproline II conformation: φ = -78°, ψ = 149°, this is ignored if --conformation_partial is not None",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=5,
        help="The cpu used in inference process",
    )
    return parser