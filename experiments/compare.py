"""Compare four fixed-filter methods on exactly the same offline clips."""
from experiments.common import evaluate, load_model, save_evaluation
from utils import common_parser, load_config


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model, bundle, checkpoint = load_model(cfg)
    methods = ("zero_filter", "environment_prior", "sample_wiener", "transformer")
    rows = evaluate(cfg, model, bundle, args.split, methods)
    save_evaluation(cfg, rows, args.split, "comparison", checkpoint)


if __name__ == "__main__":
    main()
