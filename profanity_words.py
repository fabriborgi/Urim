"""
Profanity word lists for English, Spanish, and Italian.
Each list contains common profanity/swear words that should be muted.
"""

PROFANITY_EN = {
    # Strong profanity
    "fuck", "fucking", "fucked", "fucker", "fuckers", "fucks", "motherfucker",
    "motherfucking", "motherfuckers",
    "shit", "shitting", "shitty", "bullshit", "horseshit", "dipshit",
    "bitch", "bitches", "bitching", "bitchy",
    "ass", "asshole", "assholes", "badass", "dumbass", "jackass", "smartass",
    "damn", "damned", "dammit", "goddamn", "goddammit",
    "hell", "hellhole",
    "dick", "dicks", "dickhead",
    "cock", "cocks", "cocksucker",
    "cunt", "cunts",
    "bastard", "bastards",
    "whore", "whores",
    "slut", "sluts", "slutty",
    "piss", "pissed", "pissing",
    "crap", "crappy",
    "nigga", "niggas", "nigger", "niggers",
    "retard", "retarded",
    "fag", "faggot", "faggots",
    "twat", "twats",
    "wanker", "wankers",
    "bollocks",
    "arse", "arsehole",
}

PROFANITY_ES = {
    # Spanish profanity
    "mierda", "mierdas",
    "joder", "jodido", "jodida", "jodidos", "jodidas",
    "puta", "putas", "puto", "putos", "putada",
    "coño", "coños",
    "cabrón", "cabrones", "cabrona", "cabronas",
    "culo", "culos",
    "carajo", "carajos",
    "pendejo", "pendejos", "pendeja", "pendejas",
    "chingar", "chingada", "chingado", "chingados", "chingón",
    "verga", "vergas",
    "maricón", "maricones",
    "hijo de puta", "hijueputa",
    "hostia", "hostias",
    "gilipollas",
    "mamón", "mamona",
    "idiota", "imbécil",
    "estúpido", "estúpida",
    "cojones", "cojón",
    "follar", "follado", "follada",
    "capullo", "capullos",
    "zorra", "zorras",
    "huevón", "huevona",
    "culero", "culera",
    "pinche",
    "chingón", "chingona",
    "perra", "perras",
    "mamahuevo",
    "gonorrea",
}

PROFANITY_IT = {
    # Italian profanity
    "cazzo", "cazzi", "cazzata", "cazzate",
    "minchia",
    "merda", "merde", "merdoso", "merdosa",
    "stronzo", "stronza", "stronzi", "stronze",
    "vaffanculo", "fanculo", "affanculo",
    "puttana", "puttane", "puttanata", "puttanate",
    "troia", "troie",
    "figa", "figo",
    "coglione", "coglioni", "cogliona",
    "bastardo", "bastarda", "bastardi",
    "porco", "porca",
    "madonna",
    "dio", "oddio",
    "cristo",
    "maledetto", "maledetta", "maledetti",
    "cornuto", "cornuta",
    "deficiente", "deficienti",
    "cretino", "cretina", "cretini",
    "idiota", "idioti",
    "imbecille", "imbecilli",
    "scemo", "scema", "scemi",
    "pirla",
    "frocio", "froci",
    "negro", "negra",
    "puttaniere",
    "figlio di puttana",
    "pezzo di merda",
    "testa di cazzo",
    "rompicoglioni",
    "cazzone",
}

# Combined set for quick lookup
ALL_PROFANITY = PROFANITY_EN | PROFANITY_ES | PROFANITY_IT

# Multi-word phrases that need special handling
MULTI_WORD_PROFANITY = {
    phrase for phrase in ALL_PROFANITY if " " in phrase
}

# Single word profanity
SINGLE_WORD_PROFANITY = {
    word for word in ALL_PROFANITY if " " not in word
}
