#%%
from openai import OpenAI
import json
import os
import requests

# In order to use the OpenAI API, you need to set the OPENAI_API_KEY environment variable.
# You can do this by running `export OPENAI_API_KEY="your_api_key"` in your terminal.

class AutoInterpreter:
    def __init__(self, feature_statistics, tokenizer, dataset):
        self.stats = feature_statistics
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.openai_key = os.environ["OPENAI_API_KEY"]
        self.neuronpedia_api_key = os.environ["NEURONPEDIA_API_KEY"]
        self.feature_descriptions = self.get_neuronpedia_explanations(model_id="gpt2-small", sae_id="8-res_fs49152-jb", api_key=self.neuronpedia_api_key)
        self.meta_feature_descriptions = {}

    def get_feature_summary(self, feature_idx):
        # top_words = ", ".join(list(self.stats.feature_word_count[feature_idx].keys())[:5])
        top_tokens = ", ".join(list(self.stats.feature_token_count[feature_idx].keys())[:5])
        top_examples = self.stats.get_top_examples(feature_idx, self.tokenizer, self.dataset, top_k=10, pre_context=2, post_context=1)
        examples_text = "\n".join([ex['context_text'] for ex in top_examples])
        return f"Top tokens: {top_tokens}\nExamples:\n{examples_text}"

    def get_meta_feature_summary(self, meta_feature_idx):
        features = self.stats.cluster_to_features[meta_feature_idx]
        features.sort(key=lambda x: x[1], reverse=True)
        top_features = features[:5]
        summary = ""
        for feature_idx, activation in top_features:
            feature_summary = self.get_feature_summary(feature_idx)
            neuronpedia_explanation = self.get_neuronpedia_explanation(feature_idx)
            summary += f"Feature {feature_idx} ({neuronpedia_explanation}):\n{feature_summary}\n\n"
        return summary

    def get_neuronpedia_explanation(self, feature_idx, model_id="gpt2-small", layer="8-res_fs49152-jb"):
        if feature_idx in self.feature_descriptions:
            return self.feature_descriptions[feature_idx]
        
        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{feature_idx}"
        headers = {
            "X-Api-Key": self.neuronpedia_api_key
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            feature_data = response.json()
            explanation = feature_data.get("explanations", "_")
            if explanation != "_":
                explanation = explanation[0]["description"]
            return explanation
        
        return f"Failed to fetch explanation for feature {feature_idx}"
    

    def get_neuronpedia_explanations(self, model_id, sae_id, api_key):
        url = "https://www.neuronpedia.org/api/explanation/export"
        headers = {
            "X-Api-Key": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "modelId": model_id,
            "saeId": sae_id
        }
        
        response = requests.post(url, headers=headers, json=payload)
        all_explanations = {}
        
        if response.status_code == 200:
            explanations = response.json().get("explanations", [])
            for explanation in explanations:
                all_explanations[int(explanation["index"])] = explanation["description"]
            return all_explanations

        else:
            print(f"Failed to fetch explanations. Status code: {response.status_code}")
            return {}


    def get_chatgpt_description(self, prompt):
        client = OpenAI(api_key=self.openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant that provides very short, concise, yet specific descriptions. Answser with the description only, do not include things like 'Common theme:' or 'Topic' in your answer."},
                    {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    def interpret_feature(self, feature_idx):
        if feature_idx in self.feature_descriptions:
            return self.feature_descriptions[feature_idx]
        
        summary = self.get_feature_summary(feature_idx)
        prompt = f"The following text snippets have something in common, provide a short but specific description (less than 5 words) of what this is. What specific characters/tokens/words/properties/topics/sounds do these examples have in common?:\n\n{summary}"
        description = self.get_chatgpt_description(prompt)
        self.feature_descriptions[feature_idx] = description
        return description

    def interpret_meta_feature(self, meta_feature_idx):
        if meta_feature_idx in self.meta_feature_descriptions:
            return self.meta_feature_descriptions[meta_feature_idx]
        
        summary = self.get_meta_feature_summary(meta_feature_idx)
        prompt = f"The following text snippets have something in common, provide a short but specific description (less than 5 words) of what this is. What specific characters/tokens/words/properties/topics/sounds do these examples have in common?:\n\n{summary}"
        description = self.get_chatgpt_description(prompt)
        self.meta_feature_descriptions[meta_feature_idx] = description
        return description

    def save_descriptions(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "features": self.feature_descriptions,
                "meta_features": self.meta_feature_descriptions
            }, f, indent=2)

    def load_descriptions(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.feature_descriptions = data["features"]
            self.meta_feature_descriptions = data["meta_features"]


    
if __name__ == "__main__":
    # Import necessary modules
    from feature_statistics import FeatureStatistics
    from transformer_lens import HookedTransformer
    from meta_saes.activation_store import ActivationsStore
    import os

    import sys
    sys.path.append("/workspace/Gemma/GemmaSAELens")
    from sae_lens import SAE


    # Load the model
    model = HookedTransformer.from_pretrained_no_processing("gemma-2-9b").to("cuda")

    # Load the SAE
    os.environ["GEMMA_2_SAE_WEIGHTS_ROOT"] = "/workspace/Gemma/weights/"
    assert os.path.exists(os.environ["GEMMA_2_SAE_WEIGHTS_ROOT"])
    SITE = "post_mlp_residual"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-2-saes",
        sae_id=f"30/{SITE}/131072/0_0005",
        device="cuda"
    )

    # Load the FeatureStatistics object
    stats = FeatureStatistics.load("new_features.pth", sae)

    # Load the dataset
    cfg = {
        "dataset_path": "NeelNanda/c4-10k",
        "hook_point": sae.cfg.hook_name,
        "seq_len": 1024,
        "model_batch_size": 2,
        "device": "cuda",
        "num_batches_in_buffer": 1,
        "layer": sae.cfg.hook_layer,
        "act_size": 3584,
        "batch_size": 512
    }
    activations_store = ActivationsStore(model, cfg)
    dataset = activations_store.get_complete_tokenized_dataset(add_bos=True)


    interpreter = AutoInterpreter(stats, model, dataset)
    feature_description = interpreter.interpret_feature(42)
    print(feature_description)      
    meta_feature_description = interpreter.interpret_meta_feature(7)
    print(meta_feature_description)
# interpreter.save_descriptions("feature_descriptions.json")

# %%
