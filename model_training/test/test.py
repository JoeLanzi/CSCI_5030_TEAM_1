#%% Create Test
import pandas as pd

sentences = ["May I use your pencil?",
"May I take this chair?",
"I love learning.",
"It’s as solid as a rock.",
"What a big supermarket!",
"We can offer these new products at 20% below list price.",
"He threw up all over me!",
"How big you are!",
"Keep it up!Let me see",
"Avocados are  nasty.",
"Hurry up I'm not getting any younger!",
"Is football a sport?",
"The devil made me do it!",
"I won’t say anything until you tell him.",
"What a big dog!",
"quick trip back to Tennessee for the weekend!",
"What is your favorite toy to play with or what toy do you wish you had?",
"You have a big family, right?",
"He never wants to go on a rollercoaster again!",
"That class was so hard!"]

df = pd.DataFrame(sentences,index=None)
df.columns=['Text']

language = []
for i in range(len(df)):
    language.append('English') 
df['language'] = language

df.to_csv("test.csv")
# %%
