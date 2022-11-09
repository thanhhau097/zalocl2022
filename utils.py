def clean_word(s):
    charset = '!"$\'()*+,-.:?\\_̣̀́̃’“”…'
    for c in charset:
        return s.replace(c, "")