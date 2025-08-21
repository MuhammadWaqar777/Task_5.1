from transformers import pipeline

candidate_labels = ["Network Issue","Account Problem","Password Reset","Billing Issue","Technical Support","Other"]

tickets = [
    "My internet connection keeps dropping.",
    "I forgot my login password and cannot access my account.",
    "Please change my billing address on the invoice.",
    "Laptop crashes while running software.",
    "VPN is not connecting properly."
]

zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print("Zero-Shot Results:")
for ticket in tickets:
    r = zero_shot(ticket, candidate_labels, multi_label=True)
    top3 = sorted(zip(r["labels"], r["scores"]), key=lambda x: x[1], reverse=True)[:3]
    print(ticket, [l for l, s in top3])

few_shot = pipeline("text2text-generation", model="google/flan-t5-base")

instruction = """
You are a support ticket classifier. Valid tags: Network Issue, Account Problem, Password Reset, Billing Issue, Technical Support, Other.
Return exactly three tags, comma-separated, chosen only from the valid list, ordered by relevance (most relevant first).

Examples:
Ticket: "Wi-Fi disconnects frequently."
Top 3 Tags: Network Issue, Technical Support, Other

Ticket: "I cannot log in because I forgot my password."
Top 3 Tags: Password Reset, Account Problem, Other

Ticket: "Update my payment method on file."
Top 3 Tags: Billing Issue, Account Problem, Other
"""

print("\nFew-Shot Results:")
for ticket in tickets:
    prompt = instruction + f'\nTicket: "{ticket}"\nTop 3 Tags:'
    out = few_shot(prompt, max_new_tokens=32, do_sample=False)
    text = out[0]["generated_text"].split("Top 3 Tags:")[-1].strip()
    print(ticket, text)
