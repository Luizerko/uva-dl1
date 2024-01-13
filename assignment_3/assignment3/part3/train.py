import argparse

import torch
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder


from gpt import GPT
from dataset import TextDataset
from generate import generate as generate_pretrained



class GPTLightningModule(pl.LightningModule):

    def __init__(self, config, model, train_dataset):
        super().__init__()

        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        print("running on device", self.device)

        # Unpack config hparams
        # NOTE: LearningRateFinder callback needs access to a self.lr
        self.lr = self.config.learning_rate
    
    def forward(self, x):
        # Forward pass through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        # Calculate loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        self.log('train_loss', loss, on_epoch=True)

        # Sample from predictions to calc accuracy
        acc = self.calc_accuracy_from_logits(logits, y)
        self.log('train_acc', acc, on_epoch=True)

        # Generate some sentences once in a while
        if self.global_step % self.config.generate_every_n_steps == 0:
            generated_sents = self.generate()
            self.logger.experiment.add_text('Training texts', generated_sents, self.global_step)
        return loss
    

    def calc_accuracy_from_logits(self, logits: torch.Tensor, targets: torch.Tensor):
        """ Calculates the accuracy of predictions from logits against the true targets. This function processes 
        a batch of logits (predictions before applying softmax) to calculate the accuracy of the model's predictions.
        It applies top-k filtering to the logits, computes the softmax probabilities, and then determines the top 
        predictions. The accuracy is computed by comparing these predictions with the true target values.

        Parameters:
            - logits (torch.Tensor): A tensor of logits from the model. Shape is typically (batch_size, sequence_length, vocab_size).
            - targets (torch.Tensor): The true target values. Shape is typically (batch_size, sequence_length).

        Returns:
            - torch.Tensor: The calculated accuracy as a tensor.
        """
        idx = torch.empty((targets.shape[1], 1)).to(targets.device)
        with torch.no_grad():
            for token_logits in logits:
                v, _ = torch.topk(token_logits, 50)
                token_logits[token_logits < v[:, [-1]]] = -float('Inf')
                token_probs = F.softmax(token_logits, dim=-1)
                _, idx_next = torch.topk(token_probs, k=1, dim=-1)
                idx = torch.cat((idx, idx_next), dim=1)
        idx = idx[:,1:]
        acc = torchmetrics.functional.accuracy(idx.T, targets, task='multiclass', num_classes=self.config.vocab_size)
        return acc
        

    def generate(self, prompt: str = '', num_samples: int = 5, n_steps: int = 30, do_sample: bool = True, top_k: int = 10, verbose: bool = False):
        """ Generates text based on a given prompt using either a pre-trained model or a custom-trained model. This function 
        generates text by conditioning on an input prompt. It supports both pre-trained and custom-trained models. For pre-trained models,
        it delegates to a `generate_pretrained` function. For custom-trained models, it starts from a default context or the provided prompt 
        and generates text using the model's `generate` method. For the pretrained model we use the seperate function, since we then need to 
        rely on a pretrained tokenizer of Huggingface.

        Parameters:
            - prompt (str, optional): The initial text prompt for text generation. Defaults to an empty string, which triggers a default context for custom models.
            - num_samples (int, optional): The number of text samples to generate. Only used with pre-trained models. Defaults to 5.
            - n_steps (int, optional): The number of tokens to generate. Defaults to 30.
            - do_sample (bool, optional): Whether to use sampling for text generation. Defaults to True.
            - top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to 10.
            - verbose (bool, optional): If True, enables verbose output. Currently not used in the function body. Defaults to False.

        Returns:
            - str or list of str: The generated text. If using a pre-trained model, a list of generated samples is returned. For a custom model, 
                                  a single string of generated text is returned.
        """

        if self.config.use_pretrained:
            decoded_outputs = generate_pretrained(
                prompt=prompt, 
                num_samples=num_samples,
                steps=n_steps,
                do_sample=do_sample,
                device=self.config.device,
                verbose=verbose,
            )
        else:
            context = 'Yesterday I went ' if prompt == '' else prompt
            x = torch.tensor([self.train_dataset.string_to_index[s] for s in context], dtype=torch.long)[None,...].to(self.config.device)
            y = self.model.generate(x, n_steps, temperature=1.0, do_sample=do_sample, top_k=top_k)[0]
            decoded_outputs = ''.join([self.train_dataset.index_to_string[int(i)] for i in y])
        return decoded_outputs


    def configure_optimizers(self):
        # Function to pass the optimizer to pytorch-lightning
        optimizer = self.model.configure_optimizers(self.config)
        return optimizer
    
    
    def train_dataloader(self):
        # Setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size, 
            sampler=RandomSampler(self.train_dataset, replacement=True),
            shuffle=False,
            drop_last=True, 
            pin_memory=True,
            num_workers=self.config.num_workers,
        )
        return train_loader

def train(args):
    """
    Function for training and testing a GPT model.
    Inputs:
        args (Namespace) - Namespace object from the argument parser
    """
    pl.seed_everything(args.seed)  

    # Create the dataset
    dataset = TextDataset(args.txt_file, args.txt_file, args.block_size)
    args.vocab_size = dataset.vocab_size

    # Initialise the gpt-model
    if args.use_pretrained:
        gpt_model = GPT.from_pretrained(model_type=args.model_type)
    else:
        cfg = GPT.get_default_config()
        cfg.model_type = args.model_type
        cfg.block_size = args.block_size
        cfg.vocab_size = args.vocab_size
        gpt_model = GPT(config=cfg)

    # Assuming `model` and `train_dataset` are defined and `config` is your configuration object
    lightning_model = GPTLightningModule(args, gpt_model, dataset)

    # Setup logger
    logger = TensorBoardLogger(args.log_dir, name=args.model_type)

    # Create generate callback
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="train_loss")
    lr_callback = LearningRateFinder()

    # Initialize a pytorch-lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[save_callback, lr_callback],
        max_epochs=args.num_epochs,
        accelerator=args.device,
        enable_progress_bar=args.progress_bar,
        gradient_clip_val=args.clip_grad_norm,
    )

    # Train the model
    trainer.fit(lightning_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--model_type', type=str, default='gpt-mini', help="Define the gpt2 version to be initialised")
    parser.add_argument('--block_size', type=int, default=128, help='Specify block size ')
    parser.add_argument('--use_pretrained', type=bool, default=False, help='Boolean whether to use pretrained huggingface weights.')

    # Training
    parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--generate_batch_size', type=int, default=5, help='Batch size for generated sentences in callback')
    parser.add_argument('--generate_every_n_steps', type=int, default=1000, help='Every n steps new sentences are generated by the callback')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for the optimizer (only applied on matmul weights)')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='Betas for the adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--log_dir', type=str, default='./logs', help='Sets logging directory for tensorboard logger.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')
    parser.add_argument('--num_workers', type=int, default=8, help='Num cpu workers used for training')
    parser.add_argument('--progress_bar', action='store_true', help=(
                            'Use a progress bar indicator for interactive experimentation. '
                            'Not to be used in conjuction with SLURM jobs'
                        ))

    args = parser.parse_args()
    args.device = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 

    train(args=args)