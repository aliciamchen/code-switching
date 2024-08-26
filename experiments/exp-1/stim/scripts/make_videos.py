from manim import *

class ChatBubble(Group):  # Change from VGroup to Group
    def __init__(self, message, bubble_color=BLUE, text_color=WHITE, avatar=None, **kwargs):
        super().__init__(**kwargs)

        # Create the text
        text = Text(message, font_size=24, color=text_color)

        # Create the bubble
        bubble = RoundedRectangle(
            corner_radius=0.3,
            width=text.width + 0.5,
            height=text.height + 0.3,
            fill_color=bubble_color,
            fill_opacity=1,
            stroke_color=bubble_color,
        )

        # Center the text inside the bubble
        text.move_to(bubble.get_center())

        # Add the bubble and text to the elements list
        elements = [bubble, text]

        # If avatar is provided, add it to the left of the bubble
        if avatar:
            avatar_image = ImageMobject(avatar).scale(0.5)
            avatar_image.next_to(bubble, LEFT, buff=0.1)
            elements.append(avatar_image)

        # Add all elements to the Group
        self.add(*elements)

    def position_bubble(self, align=LEFT, buff=0.5):
        # Align the bubble based on the specified side
        self.to_edge(align, buff=buff)

class ChatAnimation(Scene):
    def construct(self):
        # Create image grid on the right side and display it at the start
        image_grid = self.create_image_grid(3, 4)
        image_grid.to_edge(RIGHT, buff=0.5)
        self.add(image_grid)  # Add the image grid to the scene

        # Sample chat data for the reference game
        chat_data = [
            {"player": "aria", "text": "The tangram looks like a bird.", "time": 1.5, "role": "director", "avatar": "../identicons/blue/aria.png"},
            {"player": "katherine", "text": "Does it have wings?", "time": 3, "role": "matcher", "avatar": "../identicons/blue/katherine.png"},
            {"player": "kayla", "text": "Is it standing or flying?", "time": 4.5, "role": "matcher", "avatar": "../identicons/blue/kayla.png"},
            {"player": "oliver", "text": "I think I see it.", "time": 6, "role": "matcher", "avatar": "../identicons/blue/oliver.png"},
            {"player": "aria", "text": "Yes, it has wings.", "time": 7.5, "role": "director", "avatar": "../identicons/blue/aria.png"},
        ]

        # Display each message with a delay
        y_offset = 3
        for item in chat_data:
            text = item["text"]
            time = item["time"]
            role = item["role"]
            avatar = item["avatar"]

            # Create a chat bubble
            bubble_color = BLUE if role == "director" else GREEN
            chat_bubble = ChatBubble(text, bubble_color=bubble_color, avatar=avatar)

            # Position the bubble
            if role == "director":
                chat_bubble.position_bubble(align=LEFT)
            else:
                chat_bubble.position_bubble(align=RIGHT)

            chat_bubble.shift(UP * y_offset)
            y_offset -= 1.2

            # Animate the bubble appearing on the screen
            self.play(FadeIn(chat_bubble, shift=UP))
            self.wait(time)

        # At the end of the conversation, indicate the target tangram and chosen tangrams
        target_index = 5  # Example target index
        chosen_tangrams = {
            "katherine": 3,
            "kayla": 5,
            "oliver": 2,
        }

        self.highlight_tangrams(image_grid, target_index, chosen_tangrams)
        self.wait(2)

    def create_image_grid(self, rows, cols):
        # Load images
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

        images = [
            ImageMobject(img).scale(0.5) for img in image_paths
        ]  # Scale down to fit in grid

        # Create grid layout using Group
        grid = Group(*images).arrange_in_grid(rows=rows, cols=cols, buff=0.1)

        return grid

    def highlight_tangrams(self, image_grid, target_index, chosen_tangrams):
        # Highlight the target tangram
        target_image = image_grid[target_index]
        target_box = SurroundingRectangle(target_image, color=RED, buff=0.1)
        target_label = Text("Target", color=RED).scale(0.5).next_to(target_box, UP)
        self.play(Create(target_box), Write(target_label))

        # Highlight the chosen tangrams
        for player, index in chosen_tangrams.items():
            chosen_image = image_grid[index]
            avatar_image = ImageMobject(f"../identicons/blue/{player.lower()}.png").scale(0.3)
            avatar_image.next_to(chosen_image, DOWN, buff=0.1)
            self.play(FadeIn(avatar_image))


        # chat_data = [
        #     {"player": "Director", "text": "The tangram looks like a bird.", "time": 1.5, "role": "director", "avatar": "../identicons/blue/aria.png"},
        #     {"player": "Matcher1", "text": "Does it have wings?", "time": 3, "role": "matcher", "avatar": "../identicons/blue/katherine.png"},
        #     {"player": "Matcher2", "text": "Is it standing or flying?", "time": 4.5, "role": "matcher", "avatar": "../identicons/blue/kayla.png"},
        #     {"player": "Matcher3", "text": "I think I see it.", "time": 6, "role": "matcher", "avatar": "../identicons/blue/oliver.png"},
        #     {"player": "Director", "text": "Yes, it has wings.", "time": 7.5, "role": "director", "avatar": "../identicons/blue/aria.png"},
        # ]