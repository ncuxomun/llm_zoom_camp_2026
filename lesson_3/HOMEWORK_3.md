## Homework answers
 - **Question 1:** AI Copilot has access to current Kestra plugin documentation (because it pulls directly from Kestra docs, not from OpenAI training data)
 - **Question 2:** Vague, generic, or fabricated — the model guesses from training data. Same reason as above (i.e., training data vs 'live' information)
 - **Question 3:** 60 - 100 // With `gemini-2.5-flash` it was 64. However, I had to switch to `gemini-3.1-flash-lite` (56 tokens) because I was hitting `Free Tier` limits with the suggested (former) model. 
 - **Question 4:** 2-5x more with `gemini-3.1-flash-lite` (reason mentioned above)
 - **Question 5:** 2-4x more with `gemini-3.1-flash-lite` (reason mentioned above)
 - **Question 6:** Use traditional task-based workflows for predictability and auditability. In other words, we need a strict workflow/structure. Hence, we can't let agent have the 'freedom' per se.