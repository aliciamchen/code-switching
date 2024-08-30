import textwrap
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
AVATAR_SCALE = 0.3
AVATAR_BUFF = 0.1
CHAT_LABEL_SCALE = 0.5
CHAT_LABEL_BUFF = 0.12
BUBBLE_SHIFT = 0.3
TARGET_BOX_STROKE_WIDTH = 10
AVATAR_IMAGE_SCALE = 0.2
AVATAR_POSITION_OFFSET = 0.22
AVATAR_STACK_OFFSET = 0.4
CORRECT_GUESSES_TEXT_SCALE = 0.5
CORRECT_GUESSES_TEXT_BUFF = 0.3

class ChatBubble(Group):
    def __init__(
        self,
        message: str,
        bubble_color=BLUE,
        text_color=WHITE,
        avatar: str = None,
        role: str = "director",
        **kwargs,
    ):
        super().__init__(**kwargs)
        wrapped_text = textwrap.fill(message, width=30)
        text = Text(wrapped_text, font_size=TEXT_FONT_SIZE, color=text_color, font=TEXT_FONT)
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
            if role == "director":
                avatar_image.next_to(bubble, LEFT, buff=AVATAR_BUFF)
            else:
                avatar_image.next_to(bubble, RIGHT, buff=AVATAR_BUFF)
            elements.append(avatar_image)

        self.add(*elements)


class ChatAnimation(Scene):
    def construct(self):
        image_grid = self.create_image_grid(3, 4)
        image_grid.to_edge(RIGHT, buff=0.3)
        self.add(image_grid)

        chat_background = self.create_chat_background()
        self.add(chat_background)

        black_rectangle = self.create_black_rectangle(chat_background)
        self.add(black_rectangle)

        chat_label = self.create_chat_label(chat_background)
        self.add(chat_label)

        chat_data = self.get_chat_data()
        bubbles = []

        for item in chat_data:
            chat_bubble = self.create_chat_bubble(item)
            chat_bubble.to_edge(DOWN, buff=0.5)
            animations = [FadeIn(chat_bubble)]

            if bubbles:
                for bubble in bubbles:
                    animations.append(ApplyMethod(bubble.shift, UP * (chat_bubble.height + BUBBLE_SHIFT)))

            self.align_chat_bubble(chat_bubble, item["role"])
            bubbles.append(chat_bubble)
            self.play(AnimationGroup(*animations, lag_ratio=0))
            self.wait(item["time"])

        target_index = 5
        chosen_tangrams = {"katherine": 3, "kayla": 3, "oliver": 5}
        self.wait(2)
        self.highlight_tangrams(image_grid, target_index, chosen_tangrams)
        self.wait(2)

    def create_image_grid(self, rows: int, cols: int) -> Group:
        image_paths = [
            "../tangrams/tangram_A.png",
            "../tangrams/tangram_B.png",
            "../tangrams/tangram_C.png",
            "../tangrams/tangram_D.png",
            "../tangrams/tangram_E.png",
            "../tangrams/tangram_F.png",
            "../tangrams/tangram_G.png",
            "../tangrams/tangram_H.png",
            "../tangrams/tangram_I.png",
            "../tangrams/tangram_J.png",
            "../tangrams/tangram_K.png",
            "../tangrams/tangram_L.png",
        ]
        images = [ImageMobject(img).scale(0.4) for img in image_paths]
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
        chat_label = Text("Chat", color=BLUE, font=TEXT_FONT, weight=BOLD).scale(CHAT_LABEL_SCALE)
        chat_label.next_to(chat_background, UP, buff=CHAT_LABEL_BUFF)
        chat_label.set_z_index(2)
        return chat_label

    def get_chat_data(self) -> list:
        return [
            {"player": "aria", "text": "looks like a person sitting down", "time": 1, "role": "director", "avatar": "../identicons/blue/aria.png"},
            {"player": "katherine", "text": "to the right or to the left? ", "time": 1, "role": "matcher", "avatar": "../identicons/blue/katherine.png"},
            {"player": "kayla", "text": "to the left right? there is only one", "time": 1, "role": "matcher", "avatar": "../identicons/blue/kayla.png"},
            {"player": "oliver", "text": "I think I see it.", "time": 1, "role": "matcher", "avatar": "../identicons/blue/oliver.png"},
            {"player": "aria", "text": "yes it is sitting down to the left", "time": 2, "role": "director", "avatar": "../identicons/blue/aria.png"},
            {"player": "katherine", "text": "ok I've made my selection!", "time": 1, "role": "matcher", "avatar": "../identicons/blue/katherine.png"},
            {"player": "kayla", "text": "ok", "time": 2, "role": "matcher", "avatar": "../identicons/blue/kayla.png"},
            {"player": "aria", "text": "the one with its head down sitting to the left", "time": 1, "role": "director", "avatar": "../identicons/blue/aria.png"},
        ]

    def create_chat_bubble(self, item: dict) -> ChatBubble:
        bubble_color = DARK_BLUE if item["role"] == "director" else BLUE_B
        text_color = WHITE if item["role"] == "director" else BLACK
        return ChatBubble(
            item["text"],
            bubble_color=bubble_color,
            avatar=item["avatar"],
            role=item["role"],
            text_color=text_color,
        )

    def align_chat_bubble(self, chat_bubble: ChatBubble, role: str):
        if role != "director":
            chat_bubble.to_edge(RIGHT, buff=0.5)
            chat_bubble.shift(LEFT * 6.8)
        else:
            chat_bubble.to_edge(LEFT, buff=0.5)

    def highlight_tangrams(self, image_grid: Group, target_index: int, chosen_tangrams: dict):
        target_image = image_grid[target_index]
        target_box = SurroundingRectangle(target_image, color=GREEN_C, buff=0, stroke_width=TARGET_BOX_STROKE_WIDTH)
        target_label = Text("Target", color=GREEN_C, font=TEXT_FONT, weight=BOLD).scale(0.5).next_to(target_box, UP, buff=0.05)
        self.play(FadeIn(target_box), Write(target_label))

        avatar_positions = {}
        correct_guesses = 0

        for player, index in chosen_tangrams.items():
            chosen_image = image_grid[index]
            avatar_image = ImageMobject(f"../identicons/blue/{player.lower()}.png").scale(AVATAR_IMAGE_SCALE)

            if index not in avatar_positions:
                avatar_position = chosen_image.get_corner(UP + RIGHT) + AVATAR_POSITION_OFFSET * DOWN + AVATAR_POSITION_OFFSET * LEFT
                avatar_positions[index] = avatar_position
            else:
                avatar_position = avatar_positions[index] + AVATAR_STACK_OFFSET * DOWN
                avatar_positions[index] = avatar_position

            avatar_image.move_to(avatar_position)
            self.add(avatar_image)

            if index == target_index:
                correct_guesses += 1

        correct_guesses_text = Text(f"{correct_guesses} out of 3 guessed correctly", color=BLUE, font=TEXT_FONT, weight=BOLD).scale(CORRECT_GUESSES_TEXT_SCALE).next_to(image_grid, UP, buff=CORRECT_GUESSES_TEXT_BUFF)
        self.add(correct_guesses_text)
        self.wait(0.5)