import textwrap
import json
import random
from manim import *

# Constants
BUBBLE_CORNER_RADIUS = 0.3
BUBBLE_PADDING_WIDTH = 0.5
BUBBLE_PADDING_HEIGHT = 0.3
TEXT_FONT_SIZE = 24
TEXT_FONT = "Arial"
CHAT_BACKGROUND_WIDTH = 6.6
CHAT_BACKGROUND_HEIGHT = 7.05
BLACK_RECTANGLE_HEIGHT = 2
AVATAR_SCALE = 0.75
AVATAR_BUFF = 0.1
CHAT_LABEL_SCALE = 0.5
CHAT_LABEL_BUFF = 0.12
BUBBLE_SHIFT = 0.3
TARGET_BOX_STROKE_WIDTH = 10
AVATAR_IMAGE_SCALE = 0.5
AVATAR_POSITION_OFFSET = 0.26
AVATAR_STACK_OFFSET = 0.5
CORRECT_GUESSES_TEXT_SCALE = 0.5
CORRECT_GUESSES_TEXT_BUFF = 0.4

avatar_mappings = {
    "red": {
        "alice": "../identicons/red/HT.png",
        "bob": "../identicons/red/JW.png",
        "carol": "../identicons/red/MZ.png",
        "dave": "../identicons/red/RS.png",
    },
    "blue": {
        "alice": "../identicons/blue/AB.png",
        "bob": "../identicons/blue/LO.png",
        "carol": "../identicons/blue/PE.png",
        "dave": "../identicons/blue/WI.png",
    },
}


class ChatBubble(Group):
    def __init__(
        self,
        message: str,
        bubble_color,
        text_color=WHITE,
        avatar: str = None, # type: ignore
        role: str = "speaker",
        **kwargs,
    ):
        super().__init__(**kwargs)
        wrapped_text = textwrap.fill(message, width=30)
        text = Text(
            wrapped_text, font_size=TEXT_FONT_SIZE, color=text_color, font=TEXT_FONT
        )
        bubble = RoundedRectangle(
            corner_radius=BUBBLE_CORNER_RADIUS,
            width=text.width + BUBBLE_PADDING_WIDTH,
            height=text.height + BUBBLE_PADDING_HEIGHT,
            fill_color=bubble_color,
            fill_opacity=1,
            stroke_color=bubble_color,
        )
        text.move_to(bubble.get_center())
        elements = [bubble, text]

        if avatar:
            avatar_image = ImageMobject(avatar).scale(AVATAR_SCALE)
            if role == "speaker":
                avatar_image.next_to(bubble, LEFT, buff=AVATAR_BUFF)
            else:
                avatar_image.next_to(bubble, RIGHT, buff=AVATAR_BUFF)
            elements.append(avatar_image)

        self.add(*elements)


class ChatAnimation(Scene):
    def __init__(self, chat_data, available_tangrams, color, **kwargs):
        self.chat_data = chat_data
        self.available_tangrams = available_tangrams
        self.color = color
        super().__init__(**kwargs)

    def construct(self):
        image_grid = self.create_image_grid(2, 3)
        image_grid.to_edge(RIGHT, buff=0.3)
        self.add(image_grid)

        chat_background = self.create_chat_background()
        self.add(chat_background)

        black_rectangle = self.create_black_rectangle(chat_background)
        self.add(black_rectangle)

        chat_label = self.create_chat_label(chat_background)
        self.add(chat_label)

        bubbles = []

        for item in self.chat_data["convo"]:
            chat_bubble = self.create_chat_bubble(item)
            chat_bubble.to_edge(DOWN, buff=0.5)
            animations = [FadeIn(chat_bubble)]

            if bubbles:
                for bubble in bubbles:
                    animations.append(
                        ApplyMethod(
                            bubble.shift, UP * (chat_bubble.height + BUBBLE_SHIFT)
                        ) # type: ignore
                    )

            self.align_chat_bubble(chat_bubble, item["role"])
            bubbles.append(chat_bubble)
            self.play(AnimationGroup(*animations, lag_ratio=0))
            self.wait(item["time"])


        target = self.chat_data["target"]
        chosen_tangrams = self.chat_data["choices"]
        for player, choice in chosen_tangrams.items():
            if choice not in self.available_tangrams:
                available_tangrams_list = self.available_tangrams.copy()
                available_tangrams_list.remove(target)
                chosen_tangrams[player] = random.choice(available_tangrams_list)

        self.wait(1.5)
        self.highlight_tangrams(image_grid, target, chosen_tangrams)
        self.wait(1)

    def create_image_grid(self, rows: int, cols: int) -> Group:
        image_paths = [f"../tangrams/tangram_{tangram}.png" for tangram in self.available_tangrams]
        images = [ImageMobject(img).scale(0.55) for img in image_paths]
        return Group(*images).arrange_in_grid(rows=rows, cols=cols, buff=0.1)

    def create_chat_background(self) -> Rectangle:
        chat_background = Rectangle(
            width=CHAT_BACKGROUND_WIDTH,
            height=CHAT_BACKGROUND_HEIGHT,
            fill_color=WHITE,
            fill_opacity=0.15,
            stroke_opacity=0,
        )
        chat_background.to_edge(LEFT, buff=0.4)
        chat_background.shift(DOWN * 0.2)
        return chat_background

    def create_black_rectangle(self, chat_background: Rectangle) -> Rectangle:
        black_rectangle = Rectangle(
            width=CHAT_BACKGROUND_WIDTH,
            height=BLACK_RECTANGLE_HEIGHT,
            fill_color=BLACK,
            fill_opacity=1,
            stroke_opacity=0,
        )
        black_rectangle.next_to(chat_background, UP, buff=0)
        black_rectangle.set_z_index(1)
        return black_rectangle

    def create_chat_label(self, chat_background: Rectangle) -> Text:
        chat_label = Text("Chat", color=BLUE if self.color == "blue" else RED, font=TEXT_FONT, weight=BOLD).scale(
            CHAT_LABEL_SCALE
        )
        chat_label.next_to(chat_background, UP, buff=CHAT_LABEL_BUFF)
        chat_label.set_z_index(2)
        return chat_label

    def create_chat_bubble(self, item: dict) -> ChatBubble:
        if self.color == "red":
            bubble_color = RED_E if item["role"] == "speaker" else RED_A
            text_color = WHITE if item["role"] == "speaker" else BLACK
        elif self.color == "blue":
            bubble_color = DARK_BLUE if item["role"] == "speaker" else BLUE_B
            text_color = WHITE if item["role"] == "speaker" else BLACK
        return ChatBubble(
            item["text"],
            bubble_color=bubble_color,
            avatar=avatar_mappings[self.color][item["player"]],
            role=item["role"],
            text_color=text_color,
        )

    def align_chat_bubble(self, chat_bubble: ChatBubble, role: str):
        if role != "speaker":
            chat_bubble.to_edge(RIGHT, buff=0.5)
            chat_bubble.shift(LEFT * 6.8)
        else:
            chat_bubble.to_edge(LEFT, buff=0.5)

    def highlight_tangrams(
        self, image_grid: Group, target: str, chosen_tangrams: dict
    ):
        # target index is the index of the target in available_tangrams
        target_index = self.available_tangrams.index(target)

        target_image = image_grid[target_index]
        target_box = SurroundingRectangle(
            target_image, color=GREEN_C, buff=0, stroke_width=TARGET_BOX_STROKE_WIDTH
        )
        target_label = (
            Text("Target", color=GREEN_C, font=TEXT_FONT, weight=BOLD)
            .scale(0.5)
            .next_to(target_box, UP, buff=0.05)
        )
        self.play(FadeIn(target_box), Write(target_label))

        avatar_positions = {}
        correct_guesses = 0

        self.wait(0.5)

        for player, tangram in chosen_tangrams.items():

            index = self.available_tangrams.index(tangram)
            chosen_image = image_grid[index]
            avatar_image = ImageMobject(avatar_mappings[self.color][player]
            ).scale(AVATAR_IMAGE_SCALE)

            if index not in avatar_positions:
                avatar_position = (
                    chosen_image.get_corner(UP + RIGHT)
                    + AVATAR_POSITION_OFFSET * DOWN
                    + AVATAR_POSITION_OFFSET * LEFT
                )
                avatar_positions[index] = avatar_position
            else:
                avatar_position = avatar_positions[index] + AVATAR_STACK_OFFSET * DOWN
                avatar_positions[index] = avatar_position

            avatar_image.move_to(avatar_position)
            self.add(avatar_image)

            if index == target_index:
                correct_guesses += 1

        correct_guesses_text = (
            Text(
                f"{correct_guesses} out of 3 guessed correctly",
                color=BLUE if self.color == "blue" else RED,
                font=TEXT_FONT,
                weight=BOLD,
            )
            .scale(CORRECT_GUESSES_TEXT_SCALE)
            .next_to(image_grid, UP, buff=CORRECT_GUESSES_TEXT_BUFF)
        )
        self.add(correct_guesses_text)
        self.wait(0.5)

def generate_videos_for_item(item_number, counterbalance):
    with open(f"../items/item_{item_number}_{counterbalance}_game_info.json", "r") as f:
        game_info = json.load(f)
    available_tangrams = list(game_info["red"].keys())
    assert list(game_info["blue"].keys()) == available_tangrams

    config.media_dir = "../convo_vids"

    for color, tangram_info in game_info.items():
        for tangram, info in tangram_info.items():
            with open(f"../convos/tangram_{tangram}_game_{info["game"]}.json", "r") as f:
                convs = json.load(f)
                for conv in convs:
                    config.output_file = f"item_{item_number}_{counterbalance}_{color}_target_{conv['target']}_repNum_{conv["repNum"]}.mp4"
                    config.quality = "low_quality"
                    scene = ChatAnimation(conv, available_tangrams, color)
                    scene.render()



if __name__ == "__main__":

    generate_videos_for_item(item_number=0, counterbalance="a")
    generate_videos_for_item(item_number=0, counterbalance="b")
    generate_videos_for_item(item_number=1, counterbalance="a")
    generate_videos_for_item(item_number=1, counterbalance="b")
    generate_videos_for_item(item_number=2, counterbalance="a")
    generate_videos_for_item(item_number=2, counterbalance="b")


