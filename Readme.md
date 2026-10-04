# Hidden Risks of Jev

**An Empirical Study of Security, Privacy, and Dual Use**

Jev turns natural-language questions and application context into typed decisions and probabilities. This project studies how those decisions can be manipulated, what information they may reveal, and whether Jev can help detect unsafe content. We evaluate the official Jev service alongside NanoJev, an independent local model used for controlled training experiments.

## What the study covers

- **Jev and its workflow:** How Choice, Noul, and Score questions produce decisions that applications use to select subsequent actions.
- **Security threats:** Input manipulation through prompt injection and universal adversarial suffixes, plus training-time backdoors introduced through poisoned SFT or RLCD updates.
- **Privacy threats:** Inference of training-set membership and internal knowledge from model outputs, and inference of private attributes from application decisions.
- **Defensive use:** Detection of prompt injections, jailbreak inputs, harmful responses, and AI-generated text.
- **Discussion:** The implications of placing typed decision models in application workflows, including possible safeguards and limits of their defensive use.
