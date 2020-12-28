import math

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.sentence_prediction import SentencePredictionCriterion
from torch_scatter import scatter

from go_annotation.ontology import Ontology
ont = Ontology()

@register_criterion("go_prediction")
class GOPredictionCriterion(SentencePredictionCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super(GOPredictionCriterion, self).__init__(task, classification_head_name, regression_target)
        self._terms, self._ancestors = zip(*ont.iter_ancestor_array())  # Get node -> ancestor pairs

        head_nodes = ont.get_head_node_indices()
        self._ont_indicies = {
            'bp': ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes[0]])),
            'mf': ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes[1]])),
            'cc': ont.terms_to_indices(ont.get_descendants(ont.term_index[head_nodes[2]]))}

    def forward(self, model, sample, reduce=True):
        assert (
                hasattr(model, "classification_heads")
                and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )

        assert logits.shape[1] == ont.total_nodes, \
            f"classification head must match ontology nodes, {ont.total_nodes}"

        def convert_and_resize(x):
            return torch.tensor(x).unsqueeze(0).expand((logits.shape[0], -1))

        term_tensor = convert_and_resize(self._terms)
        ancestor_tensor = convert_and_resize(self._ancestors)

        targets = model.get_targets(sample, [logits])
        sample_size = targets.numel()

        # Normalize logits by ontology logic, requiring that child nodes have a lower score than parents
        normed_logits = scatter(torch.gather(logits, 1, term_tensor), ancestor_tensor, reduce='min')
        loss = F.binary_cross_entropy_with_logits(normed_logits, targets, reduction="sum")

        logging_output = {
            "loss": loss.data,
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        y_pred = (normed_logits > 0).to(torch.float32)
        y_true = targets

        for ont_split in ['bp', 'mf', 'cc']:
            y_pred_split = torch.gather(y_pred, -1, convert_and_resize(self._ont_indicies[ont_split]))
            y_true_split = torch.gather(y_true, -1, convert_and_resize(self._ont_indicies[ont_split]))

            logging_output[f'{ont_split}_tp'] = (y_true_split * y_pred_split).sum().to(torch.float32)
            # logging_output[f'{ont_split}_tn'] = ((1 - y_true_split) * (1 - y_pred_split)).sum().to(torch.float32)
            logging_output[f'{ont_split}_fp'] = ((1 - y_true_split) * y_pred_split).sum().to(torch.float32)
            logging_output[f'{ont_split}_fn'] = (y_true_split * (1 - y_pred_split)).sum().to(torch.float32)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, priority=5, round=3
        )

        for ont_split in ['bp', 'mf', 'cc']:
            tp = sum(log.get(f"{ont_split}_tp", 0) for log in logging_outputs)
            # tn = sum(log.get(f"{ont_split}_tn", 0) for log in logging_outputs)
            fp = sum(log.get(f"{ont_split}_fp", 0) for log in logging_outputs)
            fn = sum(log.get(f"{ont_split}_fn", 0) for log in logging_outputs)

            epsilon = 1e-7

            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)

            metrics.log_scalar(
                f"{ont_split}_f1", f1, sample_size, round=3
            )