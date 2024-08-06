from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def twitter_preprocessor(lang="Indonesia", demojize=False):
    preprocessor = TextPreProcessor(
        normalize=["url", "email", "user", "hashtag"],
        fix_text=True,
        unpack_hashtag=False,
        segmenter="corpus_id_web_wiki" if lang == "Indonesia" else "twitter_2018",
        corrector="corpus_id_web_wiki" if lang == "Indonesia" else "twitter_2018",
        demojize=demojize,
        demojize_lang="id" if lang == "Indonesia" else "en",
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
    ).pre_process_doc
    return preprocessor
