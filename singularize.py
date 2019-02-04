import re
exceptions={}
exceptions["was"]=1
exceptions["his"]=1
exceptions["its"]=1
exceptions["process"]=1
exceptions["famous"]=1
exceptions["this"]=1
exceptions["does"]=1
exceptions["has"]=1
exceptions["less"]=1
class English:
    """
    Inflector for pluralize and singularize English nouns.

    This is the default Inflector for the Inflector obj
    """

    def pluralize(self, word):
        '''Pluralizes English nouns.'''

        rules = [
            ['(?i)(quiz)$', '\\1zes'],
            ['^(?i)(ox)$', '\\1en'],
            ['(?i)([m|l])ouse$', '\\1ice'],
            ['(?i)(matr|vert|ind)ix|ex$', '\\1ices'],
            ['(?i)(x|ch|ss|sh)$', '\\1es'],
            ['(?i)([^aeiouy]|qu)ies$', '\\1y'],
            ['(?i)([^aeiouy]|qu)y$', '\\1ies'],
            ['(?i)(hive)$', '\\1s'],
            ['(?i)(?:([^f])fe|([lr])f)$', '\\1\\2ves'],
            ['(?i)sis$', 'ses'],
            ['(?i)([ti])um$', '\\1a'],
            ['(?i)(buffal|tomat)o$', '\\1oes'],
            ['(?i)(bu)s$', '\\1ses'],
            ['(?i)(alias|status)', '\\1es'],
            ['(?i)(octop|vir)us$', '\\1i'],
            ['(?i)(ax|test)is$', '\\1es'],
            ['(?i)s$', 's'],
            ['(?i)$', 's']
        ]

        uncountable_words = ['equipment', 'information', 'rice', 'money', 'species', 'series', 'fish', 'sheep', 'deer']

        irregular_words = {
            'person': 'people',
            'man': 'men',
            'child': 'children',
            'sex': 'sexes',
            'move': 'moves',
            'staff': 'staves',
        }

        lower_cased_word = word.lower();

        for uncountable_word in uncountable_words:
            if lower_cased_word[-1 * len(uncountable_word):] == uncountable_word:
                return word

        for irregular in irregular_words.keys():
            match = re.search('(' + irregular + ')$', word, re.IGNORECASE)
            if match:
                return re.sub('(?i)' + irregular + '$', match.expand('\\1')[0] + irregular_words[irregular][1:], word)

        for rule in range(len(rules)):
            match = re.search(rules[rule][0], word, re.IGNORECASE)
            if match:
                groups = match.groups()
                for k in range(0, len(groups)):
                    if groups[k] == None:
                        rules[rule][1] = rules[rule][1].replace('\\' + str(k + 1), '')

                return re.sub(rules[rule][0], rules[rule][1], word)

        return word

    def singularize(self, word):
        '''Singularizes English nouns.'''
        if word in exceptions:
            return word
        if len(word)<4:
            return word
        if word[-2:]=="ss":
            return word
        rules = [
            ['(?i)(quiz)zes$', '\\1'],
            ['(?i)(matr)ices$', '\\1ix'],
            ['(?i)(vert|ind)ices$', '\\1ex'],
            ['(?i)^(ox)en', '\\1'],
            ['(?i)(alias|status)es$', '\\1'],
            ['(?i)([octop|vir])i$', '\\1us'],
            ['(?i)(cris|ax|test)es$', '\\1is'],
            ['(?i)(shoe)s$', '\\1'],
            ['(?i)(o)es$', '\\1'],
            ['(?i)(bus)es$', '\\1'],
            ['(?i)([m|l])ice$', '\\1ouse'],
            ['(?i)(x|ch|ss|sh)es$', '\\1'],
            ['(?i)(m)ovies$', '\\1ovie'],
            ['(?i)(s)eries$', '\\1eries'],
            ['(?i)([^aeiouy]|qu)ies$', '\\1y'],
            ['(?i)([lr])ves$', '\\1f'],
            ['(?i)(tive)s$', '\\1'],
            ['(?i)(hive)s$', '\\1'],
            ['(?i)([^f])ves$', '\\1fe'],
            ['(?i)(^analy)ses$', '\\1sis'],
            ['(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$', '\\1\\2sis'],
            ['(?i)([ti])a$', '\\1um'],
            ['(?i)(n)ews$', '\\1ews'],
            ['(?i)s$', ''],
        ];

        uncountable_words = ['equipment', 'information', 'rice', 'money', 'species', 'series', 'fish', 'sheep', 'sms'];

        irregular_words = {
            'people': 'person',
            'men': 'man',
            'children': 'child',
            'sexes': 'sex',
            'moves': 'move'
        }

        lower_cased_word = word.lower();

        for uncountable_word in uncountable_words:
            if lower_cased_word[-1 * len(uncountable_word):] == uncountable_word:
                return word

        for irregular in irregular_words.keys():
            match = re.search('(' + irregular + ')$', word, re.IGNORECASE)
            if match:
                return re.sub('(?i)' + irregular + '$', match.expand('\\1')[0] + irregular_words[irregular][1:], word)

        for rule in range(len(rules)):
            match = re.search(rules[rule][0], word, re.IGNORECASE)
            if match:
                groups = match.groups()
                for k in range(0, len(groups)):
                    if groups[k] == None:
                        rules[rule][1] = rules[rule][1].replace('\\' + str(k + 1), '')

                return re.sub(rules[rule][0], rules[rule][1], word)

        return word