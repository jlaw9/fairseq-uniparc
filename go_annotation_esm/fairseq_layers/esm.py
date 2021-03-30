import logging

import torch
import torch.nn as nn
from fairseq.models import FairseqEncoder, BaseFairseqModel, register_model
from fairseq.models.masked_lm import MaskedLMModel
from fairseq.models import register_model_architecture


logger = logging.getLogger(__name__)


@register_model('esm')
class FairseqESM(MaskedLMModel):
#class FairseqESM(FairseqEncoder):

    @classmethod
    def build_model(cls, args, task):
        #torch.hub.set_dir('/tmp/.cache/torch')
        torch.hub.set_dir('/scratch/jlaw/torch/hub')
        model, alphabet = torch.hub.load("facebookresearch/esm", "esm1_t6_43M_UR50S")
        model.train()
        
        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        logger.info(args)

        #encoder = MaskedLMEncoder(args, task.dictionary)
        #return cls(args, model)
        return model

    def __init__(self, args, encoder):
        super(FairseqESM, self).__init__(args, encoder)

        #self.model = model
        #self.alphabet = alphabet


# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'rnn_classifier'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.

@register_model_architecture('esm', 'fairseq_esm')
def fairseq_esm(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    #args.hidden_dim = getattr(args, 'hidden_dim', 128)
    pass
