---
title: "insuranceLLM (LLM to generate ICD codes from clinical reports)"
date: 2023-09-14T16:05:31+05:30
draft: true
---

The goal of this project is to make a NLP model that can understand a doctor's clinical report and ouput the relevant ICD (Internationl Disease Classification) codes. These codes are then used by the insurance company to give the insurance money.
## Initial Thoughts
The initial and straight forward approch that came to our minds was that we fine tune a pre trained model to do Multi-Class Classification of the ICD codes provided we have a dataset with doctor's clinical report and the coressponding ICD codes. Unfortunately, We were not provided with such a dataset. So we brainstormed a lot and came up with 2 solutions. The major issue we thought we might face is that LLMs are not trustable. 
- **NER with calculating vector scores**
  
    This implementation focused on
    - Identifying biological entities from a clinical report using bioBERT.
    - Converting them to vectors using bioBERT embeddings.
    - Calculate the similarity score of these words with the ICD data that we already extracted ( contains ICD codes and their coresspondng description ).
    - The ICD code coressponding to the highest code will be given as the output.

    But there is an obvious flaw in this method. ICD codes have their subcodes too and their descriptions are too similar due to which the similarity scores can be too similar. Thus, even if the parent code can be identified but this fails to work to distinguish between sub ICD codes.

    ![Image alt](vector_output.png)

    Here is an example where the model finds out ICD code for 'Intestinal amebiasis'. This method doesn't always work though since bioBERT can't detect all the necessary biological entities.
  
- **An ICD fine-tuned model with Attention Manipulation**
  
  This method seems more promising than the previous one. The implementation for this goes as follows
    - First we fine-tune a pre trained model (bioGPT in our case) with the ICD guidelines PDF file.
      - Here we used the PeFT technique to train the model. Bascially, we added new parameters while freezing the pre trained ones.
      - This allowed the model to learn about the guidelines that ICD uses.
    - Next, we have to teach the models about the ICD codes. For that we again fine-tuned them using the ICD tabular PDF file.
      - Right now we are at this stage, We are still not sure if this is the way. :)
    - Recognize biological entities form the clinical report using bioBERT.
    - Then, we use Attention Manipulation to amplify the attention scores of the indentified words. This ensures that the model focuses on the right words and we get a higher chance of getting the right output.

    Here is an glimpse of how we fine-tune the model
    ```python
    import torch
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model


    class CastOutputToFloat(nn.Sequential):
        """
        A class to cast the output of the model to float32.

        parent: nn.Sequential
        return: nn.Sequential
        """
        def forward(self, x):
            return super().forward(x).to(torch.float32)


    def model_to_lora(model, config):
        """
        A function to convert the model to Lora. This function adds the Lora layer to the model. (new paremeters)

        model: model to convert
        config: LoraConfig
        return: model
        """
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        model.output_projection = CastOutputToFloat(model.output_projection)

        model = get_peft_model(model, config)
        return model


    def train(model, dataset, epochs, lr):
        """
        A function to train the model using the trainer funciton.

        model: model to train
        dataset: dataset to use
        epochs: number of epochs
        lr: learning rate
        return: trainer
        """
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=100,
                weight_decay=0.1,
                num_train_epochs=epochs,
                learning_rate=lr,
                fp16=True,
                logging_steps=1,
                output_dir="outputs",
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        return trainer
    ```
 ## Future goals
 We think the above methods can be extended in order to get a more trustable model. 
 - One such way is giving the models memory. We can do that by integrating it to a vector DB. Once a prompt is made, the model can take similar vectors form the DB and then add it to the initial prompt to get more information regarding it.
 - Reinforcement Learning is something that we can use to update the weights. But this process might take time since we also need to collect the human data.
 ## Conclusion
 We are still on the experimenting phase. We will update soon once, we get something significant.

