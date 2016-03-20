# Interpretese vs. Translationese
Code and data for the paper "Interpretese vs. Translationese: The Uniqueness of Human Strategies in Simultaneous Interpretation" published in NAACL 2016.

## Dataset
Please see README in `dat`.

## Dependencies
You need to install [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) for classification.

## Run analysis scripts
The main script is `src/count_statistics.py`.

The following arguments are required to load the data:
```
--src ja --tgt en --mt_file dat/processed/monologue.mt.en.tok.tagged --si_file dat/processed/monologue.si.en.tok.tagged --src_file dat/processed/monologue.mt.ja.tok.all --mt_align_file dat/processed/monologue.mt.align --si_align dat/processed/monologue.si.align
```

You can then specify the analysis you want to perform. For example, to compare passive voice used in Interpretese and Translationese, run
```
PYTHONPATH=./lib python src/count_statistics.py --src ja --tgt en --mt_file dat/processed/monologue.mt.en.tok.tagged --si_file dat/processed/monologue.si.en.tok.tagged --src_file dat/processed/monologue.mt.ja.tok.all --mt_align_file dat/processed/monologue.mt.align --si_align dat/processed/monologue.si.align --passive
```

Use `--help` to see other types of analysis supported.

If you use the code or the dataset, please cite
```
@inproceedings{he2016interpretese,
    title={Interpretese vs. Translationese: The Uniqueness of Human Strategies in Simultaneous Interpretation},
    author={He He and Jorday Boyd-Graber and Hal {Daum\'e III}},
    booktitle={NAACL}
    year=2016
}
```
