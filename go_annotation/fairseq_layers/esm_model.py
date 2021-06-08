import torch
from fairseq.data import Dictionary
from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from fairseq.models.roberta import RobertaModel, utils
from torch import nn


@register_model("esm")
class ESMModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        self.args = args
        #self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Add the Roberta Model arguments
        RobertaModel.add_args(parser)

        # Add the esm-specific parameters
        parser.add_argument(
            "--esm-architecture", type=str, help="ESM pretrained architecture"
        )
        # also add an inner_dim argument since for the esm model we train with a smaller inner_dim
        parser.add_argument(
            "--inner-dim", type=int, default=768,
            help="Inner dimmension to use for the GO prediction head. Default=768",
        )

    @classmethod
    def build_model(cls, args, task):

        print(f"loading '{args.esm_architecture}'")
        torch.hub.set_dir('/gpfs/alpine/scratch/jlaw/bie108/torch/hub')
        model, alphabet = torch.hub.load("facebookresearch/esm", args.esm_architecture)  # fix
        # Models are set to "eval" mode by default after loading from torch hub.
        # This changes the model to "train" mode so we can fine-tune the weights
        # UPDATE: not needed since fairseq calls model.train()
        #model.train()
        # TODO remove the unused parts of the model: contact prediction, lm_head
        # I tried deleting the contact head and lm_head, but they're both used by the esm model 
        #if args.esm_architecture == "esm1_t6_43M_UR50S":
        #    #model.contact_head = Identity()
        #    del model.contact_head
        #    #del model.embed_out
        #    #del model.embed_out_bias
        #    #print([name for name, param in list(model.named_parameters())])
        #if args.esm_architecture == "esm1_t12_85M_UR50S":
        #    del model.contact_head
        #if args.esm_architecture == "esm1b_t33_650M_UR50S":
        #    model.contact_head = Identity()
        #    model.lm_head = Identity()
            #print([name for name, param in list(model.named_parameters())])
        # check how big the model is:
        model_size = 0
        for p in model.parameters():
            model_size += p.numel() * p.element_size()
        print(f"model_size: {model_size}")
        dictionary = alphabet_to_dictionary(alphabet)
        # try updating the model max_positions
        model.args.max_positions = args.max_positions
        encoder = ESMEncoder(model, dictionary)
        assert args.max_positions <= model.args.max_positions

        return cls(args, encoder)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError


#class Identity(nn.Module):
#    def __init__(self):
#        super(Identity, self).__init__()
#        
#    def forward(self, x):
#        return x


class ESMEncoder(FairseqEncoder):
    """ESM encoder."""

    def __init__(self, model, dictionary):
        super(ESMEncoder, self).__init__(dictionary)
        self.model = model
        #self.i = 0

    def forward(
            self,
            src_tokens,
            features_only=False,
            return_all_hiddens=False,
            masked_tokens=None,
            **unused,
    ):
        assert features_only, "PSJ: lm head not currently implemented"

        result = self.model(src_tokens, repr_layers=[self.model.num_layers], return_contacts=True)
        representations = result.pop('representations')
        x = representations[self.model.num_layers]
        logits = result
        #self.i += 1
        #if self.i > 5:
        #    print(torch.cuda.memory_summary())
        #    exit()
        return x, logits


def alphabet_to_dictionary(alphabet):
    _dict = Dictionary(bos='<cls>', eos='<eos>')
    _dict.symbols = alphabet.all_toks
    _dict.indices = {token: i for token, i in enumerate(alphabet.all_toks)}
    _dict.count = [1] * len(alphabet.all_toks)

    _dict.bos_index = _dict.index('<cls>')
    _dict.pad_index = _dict.index('<pad>')
    _dict.eos_index = _dict.index('<eos>')
    _dict.unk_index = _dict.index('<unk>')

    return _dict


def base_esm_architecture(args):
    # These are just for the classification head, I don't think we use any of these options
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )


@register_model_architecture('esm', 'esm1_t6')
def esm1_t6(args):
    base_esm_architecture(args)
    # These we'll need to change for any new architecture to match the pretrained shapes.
    args.esm_architecture = getattr(args, 'esm_architecture', 'esm1_t6_43M_UR50S')
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)


@register_model_architecture('esm', 'esm1_t12')
def esm1_t12(args):
    base_esm_architecture(args)
    args.esm_architecture = getattr(args, 'esm_architecture', 'esm1_t12_85M_UR50S')
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)


@register_model_architecture('esm', 'esm1b_t33')
def esm1b_t33(args):
    base_esm_architecture(args)
    args.esm_architecture = getattr(args, 'esm_architecture', 'esm1b_t33_650M_UR50S')
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1280)


# JL: Doesn't apply to GO prediction since it works on batches of proteins
#@register_model_architecture('esm', 'esm_msa1_t12')
#def esm_msa1_t12(args):
#    base_esm_architecture(args)
#    args.esm_architecture = getattr(args, 'esm_architecture', 'esm_msa1_t12_100M_UR50S')
#    args.max_positions = getattr(args, 'max_positions', 1024)
#    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
