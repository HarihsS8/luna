"""
LUNA Quickstart Script
Run this file directly: python quickstart.py
Demonstrates:
1. Tokenization with subwords and special markers
2. Instantiation of LUNA Transformer backbone and multi-task heads
3. Mini forward pass and loss computation
4. Live inference on sentiment and toxic comment samples
"""

import torch
from luna.config import LunaConfig
from luna.tokenizer import LunaTokenizer
from luna.heads import LunaForMultiTaskClassification
from luna.eval import LunaInferencePipeline


def main():
    print("==================================================")
    print("   LUNA: Language Understanding Neural Attention   ")
    print("   BERT-like Transformer for Sentiment & Toxicity   ")
    print("==================================================")

    # 1. Initialize Configuration
    config = LunaConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    print("\n1. Initialized LUNA Architecture:")
    print(f"   - Layers: {config.num_hidden_layers}")
    print(f"   - Attention Heads: {config.num_attention_heads}")
    print(f"   - Hidden Size: {config.hidden_size}")
    print(f"   - FFN Dimension: {config.intermediate_size}")

    # 2. Tokenizer Demonstration
    tokenizer = LunaTokenizer()
    sample_text = "The camera build is stellar, but customer support was terrible!"
    encoded = tokenizer.encode(sample_text, max_length=32)
    print("\n2. Tokenization:")
    print(f"   - Raw text: '{sample_text}'")
    print(f"   - Tokens:   {encoded['tokens']}")
    print(f"   - IDs:      {encoded['input_ids'][:len(encoded['tokens'])]}")

    # 3. Model Instantiation
    model = LunaForMultiTaskClassification(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n3. Transformer Parameters: {total_params:,}")

    # 4. Forward Pass with Dummy Batch
    dummy_input = torch.tensor([encoded['input_ids']], dtype=torch.long)
    dummy_mask = torch.tensor([encoded['attention_mask']], dtype=torch.long)
    dummy_sent = torch.tensor([0], dtype=torch.long)
    dummy_toxic = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float)

    outputs = model(
        input_ids=dummy_input,
        attention_mask=dummy_mask,
        sentiment_labels=dummy_sent,
        toxic_labels=dummy_toxic,
        output_attentions=True
    )

    print("\n4. Forward Pass Verification:")
    print(f"   - Multi-Task Loss: {outputs['loss'].item():.4f}")
    print(f"   - Sentiment Logits: {outputs['sentiment_logits'].detach().numpy()[0]}")
    print(f"   - Toxic Logits:     {outputs['toxic_logits'].detach().numpy()[0]}")
    print(f"   - Attention Layers: {len(outputs['attentions'])}")

    # 5. Pipeline Demonstration
    pipeline = LunaInferencePipeline(model, tokenizer)
    test_sentences = [
        "Outstanding customer care, resolved my issue in five minutes!",
        "Shut up you absolute idiot, delete your useless account right now.",
        "The package arrived on Tuesday afternoon via standard mail.",
        "I am completely killing it at the gym today, feeling great!"
    ]

    print("\n5. Running Test Inferences:")
    for text in test_sentences:
        res = pipeline.predict(text)
        print(f"\n   [Text] '{text}'")
        print(f"   [Sentiment] {res['sentiment']['label']} (confidence: {res['sentiment']['confidence']:.2%})")
        print(f"   [Toxicity]  {'⚠️ FLAGGED TOXIC' if res['toxicity']['is_toxic'] else '✅ Clean'} (max score: {res['toxicity']['max_score']:.2f})")

    print("\n==================================================")
    print("   LUNA Transformer setup complete and verified!  ")
    print("==================================================")


if __name__ == "__main__":
    main()
