def clean_word(s):
    charset = '!"$\'()*+,-.:?\\_̣̀́̃’“”…'
    for c in charset:
        s = s.replace(c, "")
       
    return s