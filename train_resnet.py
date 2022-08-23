from trainer import create_resnet_trainer
from utils import load_config
def main():
    # Load and log experiment configuration
    config = load_config()
    print(config)
    # create trainer
    trainer = create_resnet_trainer(config)
    trainer.train()

main()