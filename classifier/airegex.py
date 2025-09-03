"""
This is a simpler regex.
"""
import re
def should_filter(text: str) -> str:
    pattern = re.compile(
        r"\b("
        r"AI"
        r"|LLM"
        r"|Artificial Intelligence"
        r"|ML"
        r")\b",
        flags=re.IGNORECASE
    )
    return "yes" if pattern.search(text) else "no"

# pattern = r"""
# (?<!\w)(?P<ai>
#   AI|IA|KI|ИИ|エーアイ|YZ|SI|UI|MI|TT|ΤΝ|TTNT|
#   Inteligencia[\s-]+Artificial|
#   Intelligence[\s-]+Artificielle|
#   Künstliche[\s-]+Intelligenz|
#   Intelligenza[\s-]+Artificiale|
#   Intelig[êe]ncia[\s-]+Artifici[áa]l|
#   Kunstmatige[\s-]+Intelligentie|
#   Искусственный[\s-]+Интеллект|
#   人工智能|
#   人工知能|
#   인공지능|
#   الذكاء[\s-]+الاصطناعي|
#   कृत्रिम[\s-]+बुद्धिमत्ता|
#   בינה[\s-]+מלאכותית|
#   Yapay[\s-]+Zek[âa]|
#   Sztuczna[\s-]+Inteligencja|
#   Umělá[\s-]+Inteligence|
#   Umelá[\s-]+Inteligencia|
#   Mesters[ée]ges[\s-]+intelligencia|
#   Tekoäly|
#   Artificiell[\s-]+Intelligens|
#   Kunstig[\s-]+Intelligens|
#   Τεχνητή[\s-]+Νοημοσύνη|
#   ปัญญาประดิษฐ์|
#   Trí[\s-]+tuệ[\s-]+nhân[\s-]+tạo
# )(?!\w)
# """
# re.compile(pattern, re.IGNORECASE | re.VERBOSE)
