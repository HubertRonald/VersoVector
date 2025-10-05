import pandas as pd

# Poemas traducidos (puedes extender esta lista)
poems = {
    "title": [
        "The Black Heralds",
        "Hope",
        "The Bread",
        "Idyll to the Fallen Child"
    ],
    "text": [
        """There are blows in life, so powerful... I don't know!
        Blows as from God's hatred; as if before them,
        the backlash of everything suffered
        were to dam up in the soul... I don't know!""",

        """Hope sleeps in a bed of clouds.
        Life’s burden drags it down,
        but it always rises again,
        trembling, like the dawn.""",

        """This bread looks at me with clean eyes.
        It speaks to me of the land that bore it
        and of the hands that kneaded it
        with love and dust.""",

        """Oh child fallen from your toy sky,
        you play now with the dust of time,
        while silence rocks you slow,
        in the lap of infinite sleep."""
    ]
}

df_vallejo = pd.DataFrame(poems)
df_vallejo.to_csv("data/vallejo_poems_en.csv", index=False)
print("✅ Archivo generado: data/vallejo_poems_en.csv")
