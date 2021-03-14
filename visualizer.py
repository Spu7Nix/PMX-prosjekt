import pygame as pg
import tensorflow_text as text
import tensorflow as tf

model = tf.keras.models.load_model('sprakmodell.h5')

# skaffer vokabularet vi har laget

bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
bert_vocab_args = dict(
    # maksimum størrelse for vokabularet
    vocab_size=8000 * 7,
    # Reserverte orddeler som må være med
    reserved_tokens=reserved_tokens,
    # flere argumenter
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)

# lager en "tokenizer", som deler tekst opp i orddeler
tokenizer = text.BertTokenizer('vocab.txt', **bert_tokenizer_params)

tokenlist = open('vocab.txt', 'r', encoding="utf-8").readlines()


# IDen til padding
PAD_ID = 0
# Maksimum lengde for vektoren
# hvis vektoren er mindre, blir det lagt til padding
max_seq_len = 20


def preprocess_bert_input(text):
    # finner IDene til alle orddelene i inputtet
    ids = tokenize_text(text, max_seq_len)
    # lager en mask, som i dette tilfettet representerer lengden på vektoren vår
    mask = tf.cast(ids > 0, tf.int64)
    mask = tf.reshape(mask, [-1, max_seq_len])
    # lager den ferdige vektoren
    # først fyller lager vi en vektor med
    # den riktige lengden (shape) fyllt med nuller
    zeros_dims = tf.stack(tf.shape(mask))
    type_ids = tf.fill(zeros_dims, PAD_ID)
    # så setter vi inn de faktiske orddelenes IDer
    type_ids = tf.cast(type_ids, tf.int64)

    return (ids, mask, type_ids)


def tokenize_text(text, seq_len):
    # bruker "tokenizeren" vi lagde tidligere til å generere tokens som passer teksten
    tokens = tokenizer.tokenize(text)
    # tilpasser outputtet
    tokens = tokens.merge_dims(1, 2)[:, :seq_len]

    # klipper vekk slutten hvis den er lenger enn maksimum lengde
    tokens = tokens[:, :seq_len]
    # legger til padding hvis den er kortere enn maksimum lengde
    tokens = tokens.to_tensor(default_value=PAD_ID)
    pad = seq_len - tf.shape(tokens)[1]
    tokens = tf.pad(tokens, [[0, 0], [0, pad]], constant_values=PAD_ID)
    return tf.reshape(tokens, [-1, seq_len])


tokens = ""
result = "hello i am result"
text = ''


def main():
    global tokens, result, text
    width, height = 640, 480
    tbw, tbh = 300, 32

    screen = pg.display.set_mode((width, height))
    font = pg.font.Font(None, 32)
    clock = pg.time.Clock()

    textbox = pg.Rect(width / 2 - tbw / 2, height * 0.1 - tbh / 2, tbw, tbh)

    color = pg.Color('white')

    done = False

    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode
                text_updated()

        screen.fill((34, 38, 54))
        # Render the current text.
        txt_surface = font.render(text, True, color)
        # Resize the box if the text is too long.
        tb_width = max(tbw, txt_surface.get_width()+10)
        textbox.w = tb_width
        textbox.x = width / 2 - tb_width / 2
        # Blit the text.
        screen.blit(txt_surface, (textbox.x+5, textbox.y+5))
        # Blit the input_box rect.
        pg.draw.rect(screen, color, textbox, 2)

        txt_surface = font.render(tokens, True, pg.Color('#50566e'))
        screen.blit(txt_surface, (width / 2 -
                                  txt_surface.get_width() / 2, height * 0.3))

        class_names = ['dansk', 'engelsk', 'spansk',
                       'japansk', 'bokmål', 'nynorsk', 'svensk']
        class_names_short = ['da', 'en', 'es', 'ja', 'nb', 'nn', 'sv']

        diagram_w = width * 0.7
        best = max(enumerate(result), key=lambda a: a[1])[0]

        for i in range(len(class_names)):
            soyle = pg.Rect(
                width / 2 - diagram_w / 2 + i *
                (diagram_w / len(class_names)) + 5,
                height - result[i] * height * 0.5,
                (diagram_w / len(class_names)) - 5,
                height
            )
            col = pg.Color("red") if i == best else color
            pg.draw.rect(screen, col, soyle, 2)

            txt_surface = font.render(class_names_short[i], True, col)
            screen.blit(txt_surface, (soyle.x + soyle.w / 2 -
                                      txt_surface.get_width() / 2, soyle.y - 50))

        pg.display.flip()
        clock.tick(30)


def text_updated():
    global tokens, result, text

    # deler inputtet inn i orddeler
    t = preprocess_bert_input(text)

    tokens = " ".join(tokenlist[id].strip() for id in t[0][0] if id != 0)
    # kjører modellen på inputtet
    result = model.predict([t])[0]


text_updated()
pg.init()
main()
pg.quit()
