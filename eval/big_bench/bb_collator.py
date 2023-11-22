import logging
import random
import string
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForBIGBench:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_definition: bool = True
    text_only: bool=False
    use_tulu_format: bool = False
    output_after_asst: bool = False
    

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
            
        for instance in batch:
            add_task_definition = self.add_task_definition

            if instance["Definition"][0] == "":
                source = instance["Instance"]["input"]
            else:
                task_input = ""
                # add the input first.
                task_input += f"{instance['Instance']['input'].strip()}"

                definition = ""
                if add_task_definition:
                    if isinstance(instance["Definition"], list):
                        definition = instance["Definition"][0].strip()
                    else:
                        definition = instance["Definition"].strip()
                    if not definition[-1] in string.punctuation:
                        definition += "."
                    definition += "\n\n"
                
                source = definition + task_input

            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                source = source
            else:
                source = self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True)

            if self.use_tulu_format:
                source = f"<|user|>\n{source}\n<|assistant|>\n"
            if self.output_after_asst:
                source = source.replace("\nOutput:\n<|assistant|>\n", "\n<|assistant|>\nOutput:")

            sources.append(source)

        if self.text_only:
            model_inputs = {"inputs": sources}
        else:
            model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)

        if "output" in batch[0]["Instance"] and batch[0]["Instance"]["output"]:
            # Randomly select one reference if multiple are provided.
            labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
            if self.text_only:
                model_inputs["labels"] = labels
            else:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        labels,
                        max_length=self.max_target_length,
                        padding=self.padding,
                        return_tensors=self.return_tensors,
                        truncation=True,
                        pad_to_multiple_of=self.pad_to_multiple_of
                    )
                label_mask = labels["attention_mask"].bool()
                model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
        else:
            model_inputs["labels"] = None

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels") and not self.text_only:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
            
        return model_inputs
