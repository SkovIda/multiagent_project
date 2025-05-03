import torch

from chessformers.chessformers.configuration import get_configuration
from chessformers.chessformers.model import Transformer
from chessformers.chessformers.tokenizer import Tokenizer

class ChessFormer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.config = get_configuration(self.cfg['config'])
        self.tokenizer = Tokenizer(self.cfg['tokenizer'])
        self.model = Transformer(self.tokenizer,
                            num_tokens=self.tokenizer.vocab_size(),
                            dim_model=self.config["model"]["dim_model"],
                            d_hid=self.config["model"]["d_hid"],
                            num_heads=self.config["model"]["num_heads"],
                            num_layers=self.config["model"]["num_layers"],
                            dropout_p=self.config["model"]["dropout_p"],
                            n_positions=self.config["model"]["n_positions"],
                            )
        self.model.load_state_dict(torch.load(self.cfg['load_model']))

        self.input_string = "<bos>"
        # self.boards = [self.input_string]
        self.prev_input_string = self.input_string

    def init(self):
        # reset model stuff: self.model.initialise()
        self.input_string = "<bos>"
        # self.boards = [self.input_string]
        self.prev_input_string = self.input_string

    def feed_info(self, move_obs) -> str:
        self.prev_input_string = self.input_string
        self.input_string += " " + move_obs
        # self.model.feedinfo(info)
        return self.input_string
        
    def get_info(self):
        try:
            self.input_string = self.model.predict(
                self.input_string, 
                stop_at_next_move=True, 
                temperature=0.2,
                )
            # self.boards.append(self.input_string)
            # print("BLACK MOVE:", self.input_string.split(" ")[-1])
        except ValueError:
            self.input_string = self.prev_input_string
            print("ILLEGAL MOVE. Please, try again.")
        except Exception as e:
            print(f"UNHANDLED EXCEPTION. Please, try again.: {e}")
        
        # return self.model.info
        return self.input_string.split(" ")[-1]
    
    def is_not_game_over(self):
        return (len(self.input_string.split(" ")) < self.config["model"]["n_positions"] 
                and self.input_string.split(" ")[-1] != self.tokenizer.eos_token)
    
    def get_attn_matrix(self):
        return self.model.get_attn_matrix(self.input_string)


if __name__ == "__main__":
    cfg_dict = {'config': "chessformers/configs/default.yaml",
                    'load_model': "chessformers/model/chessformer_epoch_13.pth",
                    'tokenizer': "chessformers/vocabs/kaggle2_vocab.txt"}
    engine = ChessFormer(cfg_dict)

    # engine.init()

    # print(
    #     "\n===== CHESSFORMERS ENGINE =====\n"
    # + "\tChessFormer self play with valid moves in PGN format.\n"
    # )

    # next_move = "e4 c5 Nf3 Nc6 d4"
    # print(f'\tUpdate hist:\t{engine.feed_info(next_move)}')


    # while engine.is_not_game_over():
    #     next_move = input("\n\nWHITE MOVE: ")
    #     print(f'\tUpdate hist:\t{engine.feed_info(next_move)}')

    #     print(f'BLACK MOVE:\t{engine.get_info()}')
    #     print(f'\tUpdate hist:\t{engine.input_string}')
        
    #     print("\n===== CHESSFORMERS ENGINE: ENCODER PARAM TEST =====")
    #     print(engine.get_attn_matrix().shape)
    #     print(engine.get_attn_matrix())
    #     # print(type(engine.model.transformer_encoder.layers[0]).self_attn)
    #     #print(engine.model.transformer_encoder.layers[0].self_attn.out_proj.weight)

    # print("--- Final board ---")
    # print(engine.input_string)

    engine.init()

    print(
        "\n===== CHESSFORMERS ENGINE =====\n"
    + "\tChessFormer self play with valid moves in PGN format.\n"
    )

    next_move = "e4 c5 Nf3 Nc6 d4"
    print(f'\tUpdate hist:\t{engine.feed_info(next_move)}')

    print(f'BLACK MOVE:\t{engine.get_info()}')
    print(f'\tUpdate hist:\t{engine.input_string}')
    attn_matrix = engine.get_attn_matrix()
    
    print("\n===== CHESSFORMERS ENGINE: ENCODER PARAM TEST =====")
    print(f'pred.shape={attn_matrix[0].shape}')
    print(f'dim of pred output: \t[{len(engine.input_string.split(" "))}\t1\t{engine.tokenizer.vocab_size()}]')
    # print(engine.get_attn_matrix())
    print(f'#self_attn layers:\t{len(attn_matrix[1])}')
    print(f'dim of model weights [out, embedding]:\t[{attn_matrix[2].shape}\t{attn_matrix[3].shape}]')
    print(f'engine.out.weight.data:\t{attn_matrix[2]}')
    print(f'engine.embedding.weight.data:\t{attn_matrix[3]}')
    print(f'word_weights (dim={attn_matrix[4].shape}):\t{attn_matrix[4]}')
    print(f'word_idx:\t{attn_matrix[5]}')


