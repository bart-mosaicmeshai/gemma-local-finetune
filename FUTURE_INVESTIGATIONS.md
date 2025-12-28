# Future Investigation Ideas

This document captures potential investigations and improvements for the gemma-local-finetune project.

## 270M Model Coherence Investigation

**Observation:** The fine-tuned 270M Bluey model shows significant coherence issues in browser deployment (repetitive loops, word salad), while other browser-based demos using the base Gemma 270M model (e.g., [Bedtime Story Generator](https://huggingface.co/spaces/webml-community/bedtime-story-generator)) produce coherent output.

**Hypothesis:** The coherence degradation may be caused by the fine-tuning process rather than the model size or browser deployment method.

**Potential causes to investigate:**
1. **Training data size**: 111 examples may be insufficient for maintaining coherence
2. **Overfitting to short responses**: Model learned to generate short, truncated responses (see Part 4 early stopping issue)
3. **Response length patterns**: Training data averaged 52-76 words - model may have learned this constraint too rigidly
4. **Base model differences**: Verify which specific 270M checkpoint was used vs other demos
5. **Generation parameters**: Compare temperature, top_p, repetition_penalty settings with working demos

**Investigation approach:**
1. Deploy base (non-fine-tuned) google/gemma-3-270m-it model to browser
2. Compare coherence between base and fine-tuned models with identical prompts
3. Test fine-tuned model with various generation parameter configurations
4. Analyze training data characteristics (length distribution, repetition patterns)
5. Consider fine-tuning with larger, more diverse dataset (500+ examples)

**Expected outcome:** Isolate whether coherence issues stem from fine-tuning approach, training data quality, or generation parameters. Document findings in follow-up blog post if investigation is pursued.

**Related:** Blog series Part 7 documents browser deployment but doesn't investigate coherence issues in depth.
