# N_gram

## test runs
|tokenization_level| n_tested | best_n | Dev set WER |
| :--------------- | :------: |:-----: | ----: |
| word             | 1-5    | 3 | 0.12026726057906459|
| subword          | 1-6    | 5 | 0.11804008908685969| 
| char             | 1-6   | 5 | 0.0645879732739421 |



transformer: 0.18505
python3 n_gram.py --n=4 -t=subword -e=t -sm=linearinterpolation
Model type:  n_gram
Loading data...
Data loaded and cleaned.
Running GPT-2 tokenizer
vocab_size 26933
~ initializing and training NGramLM model..
~ initializing n-gram model with n=4 and sm=linearinterpolation
~ evaluating model perplexity..
Model perplexity on the val set: 3.5028913598390923
Model perplexity on the dev set: 3.4922647992829856
Model perplexity on the test set: 3.481686412815796
~ evaluating model wer..
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Dev set WER was:  0.09131403118040089

tested wer
char_6 = 16904
char_5_linear_smooth = 16014
**char_5_smooth = .13523**
char_5 = .16014
char_? = .16726
char n =6 no sm =.17438

**sub_3_linearinterpolation =.17438 **
Model perplexity on the val set: 5.029838774284398
Model perplexity on the dev set: 5.0038581891137275
Model perplexity on the test set: 5.000632834371813
Dev set WER was:  0.08908685968819599

subword_3_smooth = 18149
subword_4_smooth = 18327
**sub_4 = .18149**
sub_5 = .18505

word_5 = .19217
word_3 = .19039

word, n >=3 same

subword, n = [1,4] more n, lower perplexity, higher wer

char, n=6  0.0935412026726058
