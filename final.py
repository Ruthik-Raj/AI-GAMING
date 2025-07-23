from deepface import DeepFace
import pyttsx3
import random
import time
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import math
import numpy as np


class MomoGame:
    def __init__(self):
        # === MediaPipe Face Mesh Setup ===
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # === MediaPipe Pose Setup for Dance Analysis ===
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Increased for better accuracy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # === MediaPipe Hands Setup ===
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

        # Blink detection parameters
        self.EAR_THRESHOLD = 0.21
        self.CONSEC_FRAMES = 3

        # === FULL LANDMARK TRACKING ===
        # ALL 33 Body Pose Landmarks
        self.ALL_POSE_LANDMARKS = list(range(33))

        # Key facial landmarks for expression tracking
        self.FACE_LANDMARKS = [
            0,  # nose tip
            1, 2,  # inner eyes
            4, 5,  # outer eyes
            7, 8,  # ears
            9, 10,  # mouth corners
        ]

        # Hand landmarks (21 points per hand)
        self.HAND_LANDMARKS = list(range(21))

        # Categorized landmarks for detailed analysis
        self.HEAD_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Head and face
        self.TORSO_LANDMARKS = [11, 12, 23, 24]  # Shoulders and hips
        self.ARM_LANDMARKS = [11, 12, 13, 14, 15, 16]  # Shoulders to wrists
        self.LEG_LANDMARKS = [23, 24, 25, 26, 27, 28]  # Hips to ankles
        self.FOOT_LANDMARKS = [29, 30, 31, 32]  # Feet details

        # Joke collection
        self.jokes = [
            "What kind of music do balloons hate? Pop music.",
            "What's a skeleton's least favorite room? The living room.",
            "Why was the math book sad? It had too many problems.",
            "What kind of tree fits in your hand? A palm tree.",
            "What do you call a flying bagel? A plain bagel.",
            "What kind of key opens a banana? A monkey.",
            "Why don't eggs tell jokes? Because they might crack up.",
            "Why did the coffee file a police report? It got mugged.",
            "What did one ocean say to the other? Nothing, they just waved.",
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "What do you call a dinosaur that crashes his car? Tyrannosaurus Wrecks!",
            "Why don't skeletons fight each other? They don't have the guts!"
        ]

        # Dance moves for SipSync
        self.moves = [
            "do the chicken dance with finger guns",
            "strike a yoga pose while nodding your head",
            "wave your arms like spaghetti and wiggle your fingers",
            "do your worst impression of a T-Rex with tiny hand movements",
            "moonwalk while pointing at the sky",
            "robot dance with precise finger movements",
            "floss dance with exaggerated facial expressions",
            "disco point while winking",
            "jazz hands with head bobbing",
            "air guitar with rock face expression"
        ]

        # === NEW: Riddle collection for Riddle Rush ===
        self.riddles = [
            {
                "question": "I have keys but no locks. I have space but no room. You can enter but not go inside. What am I?",
                "answer": "keyboard"},
            {"question": "I'm tall when I'm young and short when I'm old. What am I?", "answer": "candle"},
            {"question": "I have a face but no eyes, hands but no arms. What am I?", "answer": "clock"},
            {"question": "I have teeth but cannot bite. What am I?", "answer": "comb"},
            {"question": "I go up but never come down. What am I?", "answer": "stairs"},
            {"question": "I have a neck but no head, and wear a cap. What am I?", "answer": "bottle"},
            {"question": "I'm full of holes but still hold water. What am I?", "answer": "sponge"},
            {"question": "I have legs but cannot walk. What am I?", "answer": "table"},
            {"question": "I have an eye but cannot see. What am I?", "answer": "needle"},
            {"question": "I'm light as a feather, but even the strongest person can't hold me for long. What am I?",
             "answer": "breath"},
            {"question": "I have wings but cannot fly. I have a body but no head. What am I?", "answer": "airplane"},
            {"question": "I'm always hungry and must be fed. The finger I touch will soon turn red. What am I?",
             "answer": "fire"},
            {"question": "I have cities but no houses, forests but no trees, water but no fish. What am I?",
             "answer": "map"},
            {"question": "I can be cracked, made, told, and played. What am I?", "answer": "joke"},
            {"question": "I get wet while drying. What am I?", "answer": "towel"},
            {"question": "I have a golden head and a golden tail, but no body. What am I?", "answer": "coin"},
            {"question": "I'm black when clean and white when dirty. What am I?", "answer": "chalkboard"},
            {"question": "I have four wheels and flies, but I'm not alive. What am I?", "answer": "garbage truck"},
            {"question": "I have a spine but no bones. What am I?", "answer": "book"},
            {"question": "I'm taken before you get me. What am I?", "answer": "photograph"},
            {"question": "I can travel around the world while staying in a corner. What am I?", "answer": "stamp"},
            {"question": "I'm always in front of you but can't be seen. What am I?", "answer": "future"},
            {"question": "I have a mouth but never eat, a bed but never sleep. What am I?", "answer": "river"},
            {"question": "I'm lighter than air but a hundred people cannot lift me. What am I?", "answer": "bubble"},
            {"question": "I have branches but no fruit, trunk but no bark. What am I?", "answer": "bank"},
            {"question": "I can be long or short, grown or bought, painted or left bare. What am I?",
             "answer": "fingernails"},
            {"question": "I have a heart that doesn't beat. What am I?", "answer": "artichoke"},
            {"question": "I'm always coming but never arrive. What am I?", "answer": "tomorrow"},
            {"question": "I have ears but cannot hear. What am I?", "answer": "corn"},
            {"question": "I can be hot or cold, served in a bowl, and I'm good for the soul. What am I?",
             "answer": "soup"},
            {"question": "I have a ring but no finger. What am I?", "answer": "telephone"},
            {
                "question": "I'm round like a ball but not quite as round. I'm yellow like the sun but not quite as bright. What am I?",
                "answer": "lemon"},
            {"question": "I have a sole but I'm not a shoe. What am I?", "answer": "fish"},
            {"question": "I can be smooth or rough, big or small, and I'm found at the beach. What am I?",
             "answer": "stone"},
            {"question": "I have strings but I'm not a guitar. I can fly but I'm not a bird. What am I?",
             "answer": "kite"},
            {"question": "I'm white when dirty and clear when clean. What am I?", "answer": "window"},
            {"question": "I have a cap but no head, a stem but no leaves. What am I?", "answer": "mushroom"},
            {"question": "I can be shuffled, dealt, and played, but I'm not music. What am I?", "answer": "cards"},
            {"question": "I have teeth on my edge but cannot chew. What am I?", "answer": "saw"},
            {"question": "I'm full during the day and empty at night. What am I?", "answer": "shoes"},
            {"question": "I have a tongue but cannot taste. What am I?", "answer": "shoe"},
            {"question": "I can be folded but I'm not clothes. I can be written on but I'm not a board. What am I?",
             "answer": "paper"},
            {"question": "I have a blade but I'm not a knife. I spin but I'm not a wheel. What am I?", "answer": "fan"},
            {"question": "I'm pushed and pulled but never move. What am I?", "answer": "door"},
            {"question": "I have a screen but show no movies. I have keys but open no doors. What am I?",
             "answer": "computer"},
            {"question": "I can be wound up but I'm not a toy. I tell time but I'm not digital. What am I?",
             "answer": "watch"},
            {"question": "I have a point but I'm not sharp. I can write but I'm not a pen. What am I?",
             "answer": "pencil"},
            {"question": "I have a head, a foot, and four legs, but I'm not alive. What am I?", "answer": "bed"}
        ]

    # === Text-to-Speech Methods ===
    def speak(self, text, rate=190):
        """Convert text to speech with female voice preference"""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            # Try to find a female voice
            for voice in voices:
                if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break

            engine.setProperty('rate', rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")

    # === Computer Vision Helper Methods ===
    def euclidean_distance(self, p1, p2):
        """Calculate distance between two MediaPipe landmarks"""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def euclidean_distance_3d(self, p1, p2):
        """Calculate 3D distance between two MediaPipe landmarks"""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def calculate_pose_similarity(self, poses1, poses2, hands1=None, hands2=None, faces1=None, faces2=None):
        """
        Full similarity calculation including:
        - All 33 body pose landmarks
        - Hand landmarks (if available)
        - Facial expression landmarks (if available)
        """
        if not poses1 or not poses2:
            return 0.0

        similarities = []
        min_length = min(len(poses1), len(poses2))

        for i in range(min_length):
            pose1 = poses1[i]
            pose2 = poses2[i]

            if pose1 and pose2:
                total_similarity = 0
                total_weight = 0

                # === BODY POSE COMPARISON (All 33 landmarks) ===
                body_similarity = self._compare_body_landmarks(pose1, pose2)
                total_similarity += body_similarity * 0.6  # 60% weight for body
                total_weight += 0.6

                # === HAND COMPARISON ===
                if hands1 and hands2 and i < len(hands1) and i < len(hands2):
                    hand_similarity = self._compare_hand_landmarks(hands1[i], hands2[i])
                    total_similarity += hand_similarity * 0.25  # 25% weight for hands
                    total_weight += 0.25

                # === FACE COMPARISON ===
                if faces1 and faces2 and i < len(faces1) and i < len(faces2):
                    face_similarity = self._compare_face_landmarks(faces1[i], faces2[i])
                    total_similarity += face_similarity * 0.15  # 15% weight for face
                    total_weight += 0.15

                if total_weight > 0:
                    similarities.append(total_similarity / total_weight)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _compare_body_landmarks(self, pose1, pose2):
        """Compare all 33 body pose landmarks with different weights"""
        frame_similarity = 0
        valid_landmarks = 0

        # Different weights for different body parts
        landmark_weights = {
            # Head/Face (high weight for expression)
            **{i: 1.5 for i in self.HEAD_LANDMARKS},
            # Arms (high weight for gesture)
            **{i: 1.3 for i in self.ARM_LANDMARKS},
            # Torso (medium weight for posture)
            **{i: 1.0 for i in self.TORSO_LANDMARKS},
            # Legs (medium weight for stance)
            **{i: 1.0 for i in self.LEG_LANDMARKS},
            # Feet (lower weight)
            **{i: 0.8 for i in self.FOOT_LANDMARKS},
        }

        for landmark_idx in self.ALL_POSE_LANDMARKS:
            if (landmark_idx < len(pose1.landmark) and
                    landmark_idx < len(pose2.landmark)):

                lm1 = pose1.landmark[landmark_idx]
                lm2 = pose2.landmark[landmark_idx]

                # Only compare if both landmarks are visible
                if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                    # Use 3D distance for better accuracy
                    distance = self.euclidean_distance_3d(lm1, lm2)
                    # Convert distance to similarity
                    similarity = max(0, 1 - (distance * 3))  # Adjusted scale

                    # Apply landmark-specific weight
                    weight = landmark_weights.get(landmark_idx, 1.0)
                    frame_similarity += similarity * weight
                    valid_landmarks += weight

        return frame_similarity / valid_landmarks if valid_landmarks > 0 else 0.0

    def _compare_hand_landmarks(self, hands1, hands2):
        """Compare hand landmarks between two frames"""
        if not hands1 or not hands2:
            return 0.0

        # Compare each detected hand
        hand_similarities = []

        for h1 in hands1:
            best_match = 0
            for h2 in hands2:
                similarity = self._compare_single_hand(h1, h2)
                best_match = max(best_match, similarity)
            hand_similarities.append(best_match)

        return sum(hand_similarities) / len(hand_similarities) if hand_similarities else 0.0

    def _compare_single_hand(self, hand1, hand2):
        """Compare landmarks of a single hand"""
        if not hand1 or not hand2:
            return 0.0

        similarities = []

        # Compare all 21 hand landmarks
        for i in range(min(len(hand1.landmark), len(hand2.landmark))):
            lm1 = hand1.landmark[i]
            lm2 = hand2.landmark[i]

            distance = self.euclidean_distance_3d(lm1, lm2)
            similarity = max(0, 1 - (distance * 4))  # Hands are more precise
            similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _compare_face_landmarks(self, face1, face2):
        """Compare facial landmarks for expression matching"""
        if not face1 or not face2:
            return 0.0

        # Focus on key facial expression points
        key_face_points = [
            # Eyebrows
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            # Eyes
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            # Nose
            1, 2, 5, 4, 6, 19, 20, 94, 125,
            # Mouth
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
        ]

        similarities = []

        for point_idx in key_face_points:
            if (point_idx < len(face1.landmark) and point_idx < len(face2.landmark)):
                lm1 = face1.landmark[point_idx]
                lm2 = face2.landmark[point_idx]

                distance = self.euclidean_distance(lm1, lm2)
                similarity = max(0, 1 - (distance * 8))  # Face landmarks are very precise
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def extract_all_features(self, frame):
        """Extract pose, hand, and face landmarks from a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract pose
        pose_results = self.pose.process(rgb_frame)
        pose_landmarks = pose_results.pose_landmarks if pose_results.pose_landmarks else None

        # Extract hands
        hand_results = self.hands.process(rgb_frame)
        hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

        # Extract face
        face_results = self.face_mesh.process(rgb_frame)
        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None

        return pose_landmarks, hand_landmarks, face_landmarks

    def extract_pose_features(self, frame):
        """Extract pose landmarks from a frame (backward compatibility)"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks if results.pose_landmarks else None

    def eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Vertical distances
        A = self.euclidean_distance(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        B = self.euclidean_distance(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        # Horizontal distance
        C = self.euclidean_distance(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

        ear = (A + B) / (2.0 * C)
        return ear

    # === SIMPLIFIED EMOTION DETECTION (from second code) ===
    def detect_emotion(self, frame):
        """Detect dominant emotion from face in frame - SIMPLIFIED VERSION"""
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            return emotion
        except Exception as e:
            print(f"⚠️ Emotion detection error: {e}")
            return "neutral"

    # === Game-Specific Methods ===
    def deliver_joke(self, joke):
        """Split and deliver joke with dramatic timing"""
        if "?" in joke:
            question, punchline = joke.split("?", 1)
            question += "?"
            punchline = punchline.strip()

            print(f"\n🎤 Momo asks: {question}")
            self.speak(question, rate=170)
            time.sleep(3)  # Dramatic pause
            print(f"🤣 Momo answers: {punchline}")
            self.speak(punchline, rate=170)
        else:
            print(f"\n🎤 Momo says: {joke}")
            self.speak(joke)

    # === NEW GAME 4: Riddle Rush ===
    def play_riddle_rush(self, cap):
        """Challenge player to solve riddles in 10 seconds - 2 rounds, need 1 correct to avoid drinking"""
        self.speak("Welcome to Riddle Rush! ")


        correct_answers = 0
        used_riddles = []

        for round_num in range(1, 3):
            print(f"\n🎯 ROUND {round_num} of 2")
            self.speak(f"Round {round_num}! Get ready for your riddle!", rate=200)

            # Select a random riddle that hasn't been used
            available_riddles = [r for r in self.riddles if r not in used_riddles]
            riddle = random.choice(available_riddles)
            used_riddles.append(riddle)

            print(f"\n🧩 RIDDLE: {riddle['question']}")
            self.speak(riddle['question'], rate=180)

            print("💭 You have 10 seconds to think!")

            # 10-second countdown without camera
            print("\n⏰ COUNTDOWN:")
            for countdown in range(10, 0, -1):
                print(f"  {countdown}...")
                time.sleep(1)

            # Get user's answer immediately
            print(f"\n⏰ Time's up! What's your answer?")
            self.speak("Time's up! What's your answer?")
            user_answer = input("👉 Your answer: ").strip().lower()

            # Check answer - improved logic
            correct_answer = riddle['answer'].lower().strip()
            user_answer_clean = user_answer.strip()

            # Multiple ways to check if answer is correct
            is_correct = (
                    user_answer_clean == correct_answer or  # Exact match
                    user_answer_clean in correct_answer or  # Partial match (like "breath" in "breath")
                    correct_answer in user_answer_clean or  # Answer contains correct word
                    any(word.strip() == user_answer_clean for word in correct_answer.split())  # Word-by-word check
            )

            if is_correct:
                print(f"✅ CORRECT! The answer was '{riddle['answer']}'")
                self.speak(f"Correct! The answer was {riddle['answer']}. Well done!")
                correct_answers += 1
            else:
                print(f"❌ WRONG! The correct answer was '{riddle['answer']}'")
                print(f"   You answered: '{user_answer_clean}' | Expected: '{correct_answer}'")
                self.speak(f"Nope! The answer was {riddle['answer']}. Better luck next round!")

            # Show score so far
            print(f"📊 Score so far: {correct_answers}/{round_num} correct")

            # Short break between rounds
            if round_num < 2:
                print("⏱️ 3 second break...")
                time.sleep(3)

        # Final results
        print(f"\n🏆 RIDDLE RUSH FINAL RESULTS:")
        print(f"Correct Answers: {correct_answers}/2")

        if correct_answers >= 1:
            result_msg = f"You got {correct_answers} out of 2 correct! Your brain saved you - no drink needed!"
            print(f"🧠 {result_msg}")
            self.speak(result_msg)
        else:
            result_msg = "You got 0 out of 2 correct! Time to drink and think about your life choices!"
            print(f"🍺 {result_msg}")
            self.speak(result_msg)

        # Add some Momo commentary
        if correct_answers == 2:
            bonus_msg = "Perfect score! Are you secretly a riddle master?"
            print(f"🤯 {bonus_msg}")
            self.speak(bonus_msg)
        elif correct_answers == 1:
            bonus_msg = "Just barely saved yourself! One more wrong and you'd be drinking!"
            print(f"😅 {bonus_msg}")
            self.speak(bonus_msg)
        else:
            bonus_msg = "My AI brain is disappointed in your human brain!"
            print(f"🤖 {bonus_msg}")
            self.speak(bonus_msg)

    # === GAME 1: Face-Off (SIMPLIFIED from second code) ===
    def play_face_off(self, cap):
        """Challenge player to keep a straight face during joke - 3 rounds - SIMPLIFIED VERSION"""
        self.speak("Get ready for three rounds of straight-face madness!")

        wins = 0
        losses = 0
        self.speak("Starting in 3... 2... 1... Go!")

        for round_num in range(1, 4):
            print(f"\n🎯 ROUND {round_num} of 3")
            self.speak(f"Round {round_num}! Keep that poker face!", rate=200)

            joke = random.choice(self.jokes)
            self.deliver_joke(joke)

            reaction_detected = False
            start_time = time.time()

            self.speak(" Momo is watching your face...")

            while time.time() - start_time < 8:  # 8 second window
                ret, frame = cap.read()
                if not ret:
                    continue

                # SIMPLIFIED emotion detection (from second code)
                emotion = self.detect_emotion(frame)

                # Display current emotion (optional debug)
                if time.time() - start_time > 2:  # Start checking after joke delivery
                    print(f"Good")

                    # SIMPLIFIED detection - only check for 'happy' (from second code)
                    if emotion in ['happy']:
                        msg = f"Round {round_num}: You cracked! That's a drink!"
                        print(f"😂 {msg}")
                        self.speak(msg)
                        reaction_detected = True
                        losses += 1
                        break

                # Show video feed
                cv2.putText(frame, f'Round {round_num} - {emotion}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Face-Off", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

            if not reaction_detected:
                msg = f"Round {round_num}: Stone-cold! You survived this one!"
                print(f"😐 {msg}")
                self.speak(msg)
                wins += 1

            # Short break between rounds
            if round_num < 3:
                print("⏱️ 3 second break...")
                time.sleep(3)

        cv2.destroyAllWindows()

        # Final results
        print(f"\n🏆 FINAL RESULTS:")
        print(f"Wins: {wins} | Losses: {losses}")

        if wins > losses:
            result_msg = f"You won {wins} out of 3 rounds! Impressive poker face! But take a victory sip anyway!"
        elif losses > wins:
            result_msg = f"You only survived {wins} out of 3 rounds. Time to drink up, human!"
        else:
            result_msg = "A tie! That means everyone drinks!"

        print(f"🎉 {result_msg}")
        self.speak(result_msg)

    # === GAME 2: Blink Challenge ===
    def play_blink_challenge(self, cap):
        """Challenge player not to blink"""
        print("\n👀 Starting Blink Challenge!")
        self.speak("Stare into my soul without blinking. I can now track your entire face!")

        blink_counter = 0
        frame_counter = 0
        blink_detected = False
        start_time = time.time()
        challenge_duration = 15  # 15 seconds

        while time.time() - start_time < challenge_duration:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_ear = self.eye_aspect_ratio(landmarks, self.LEFT_EYE_IDX)
                right_ear = self.eye_aspect_ratio(landmarks, self.RIGHT_EYE_IDX)
                avg_ear = (left_ear + right_ear) / 2.0

                # Display
                cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Blinks: {blink_counter}', (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Show remaining time
                remaining = int(challenge_duration - (time.time() - start_time))
                cv2.putText(frame, f'Time: {remaining}s', (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Draw face mesh for better visualization
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.multi_face_landmarks[0],
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

                # Blink detection logic
                if avg_ear < self.EAR_THRESHOLD:
                    frame_counter += 1
                else:
                    if frame_counter >= self.CONSEC_FRAMES:
                        blink_counter += 1
                        print("👁 Blink Detected! You lose!")
                        self.speak("Caught you blinking! Drink up!")
                        blink_detected = True
                        cv2.putText(frame, "BLINK DETECTED!", (200, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        cv2.imshow("Blink Challenge", frame)
                        cv2.waitKey(2000)
                        break
                    frame_counter = 0

            cv2.imshow("Blink Challenge", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        cv2.destroyAllWindows()

        if not blink_detected:
            print("✅ Impressive! You didn't blink. Everyone else drinks!")
            self.speak("Impressive! You have robot eyes. Everyone else drinks!")

    # === GAME 3: SipSync with Full Body Tracking ===
    def play_sipsync(self, cap):
        """Dance battle with full body, hand, and face tracking"""
        print("\n🕺 Starting SipSync Challenge!")
        self.speak("Time for the ultimate dance battle with full body tracking!")

        # Get player names
        players = []
        roles = ["dance leader", "move copier"]

        for i in range(2):
            self.speak(f"What's player {i + 1}'s name? They will be the {roles[i]}!")
            name = input(f"\n👉 Player {i + 1} name ({roles[i]}): ").strip()
            players.append(name if name else f"Player {i + 1}")

        print(f"\n🎭 Dancers: {players[0]} vs {players[1]}")
        self.speak(f"Tonight's dance battle: {players[0]} versus {players[1]}!")

        # Randomly select dance leader
        leader = random.choice(players)
        follower = players[1] if players[0] == leader else players[0]

        print(f"\n👑 {leader} is the dance leader!")
        self.speak(f"{leader}, you're the leader! Show us your movesssssssss!")

        # Give leader time to think
        self.speak("Take 5 seconds to think of an awesome dance move!")
        time.sleep(5)

        # Leader performs (10 seconds) - FULL TRACKING
        print(f"\n🎬 {leader}, show your move! You have 10 seconds!")

        leader_start = time.time()
        leader_poses = []
        leader_hands = []
        leader_faces = []

        while time.time() - leader_start < 10:
            ret, frame = cap.read()
            if not ret:
                continue

            # Extract ALL features
            pose_landmarks, hand_landmarks, face_landmarks = self.extract_all_features(frame)

            if pose_landmarks:
                leader_poses.append(pose_landmarks)
            leader_hands.append(hand_landmarks)
            leader_faces.append(face_landmarks)

            # Visualization
            remaining = int(10 - (time.time() - leader_start))
            cv2.putText(frame, f"{leader}'s Turn: {remaining}s", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "FULL BODY TRACKING!", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Draw all landmarks
            if pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if hand_landmarks:
                for hand in hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
            if face_landmarks:
                self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)

            cv2.imshow("SipSync - Leader", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.speak("Time's up! That was incredible!")

        # Follower performs (10 seconds) - FULL TRACKING
        print(f"\n🎯 Now {follower}, copy EVERYTHING - body, hands, and face!")
        self.speak(f"{follower}, show us your impression of the entire performance!", rate=200)

        follower_start = time.time()
        follower_poses = []
        follower_hands = []
        follower_faces = []

        while time.time() - follower_start < 10:
            ret, frame = cap.read()
            if not ret:
                continue

            # Extract ALL features
            pose_landmarks, hand_landmarks, face_landmarks = self.extract_all_features(frame)

            if pose_landmarks:
                follower_poses.append(pose_landmarks)
            follower_hands.append(hand_landmarks)
            follower_faces.append(face_landmarks)

            # Visualization
            remaining = int(10 - (time.time() - follower_start))
            cv2.putText(frame, f"{follower}'s Turn: {remaining}s", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "COPY EVERYTHING!", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Draw all landmarks
            if pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if hand_landmarks:
                for hand in hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
            if face_landmarks:
                self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)

            cv2.imshow("SipSync - Follower", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
        self.speak("Analysis time!")

        # AI analysis
        print(f"\n⚖️ Momo's AI is analyzing...")
        self.speak("Let me analyze your full-body performance with my advanced AI!")
        time.sleep(2)

        # Calculate similarity
        similarity_score = self.calculate_pose_similarity(
            leader_poses, follower_poses, leader_hands, follower_hands, leader_faces, follower_faces
        )
        print(f"🧠 Similarity Score: {similarity_score:.3f}")

        # Calculate individual metrics
        leader_metrics = self.analyze_dance_quality(leader_poses, leader_hands, leader_faces)
        follower_metrics = self.analyze_dance_quality(follower_poses, follower_hands, follower_faces)

        # ✅ CORRECTED AI JUDGMENT - GOOD = NO DRINK, BAD = DRINK
        if similarity_score >= 0.85:
            print(f"🎉 MOMO'S VERDICT: {follower} PERFECTLY matched body, hands, AND face!")
            self.speak(f"INCREDIBLE! {follower} copied everything perfectly! No drinks needed - you nailed it!")

        elif similarity_score >= 0.70:
            print(f"👏 MOMO'S VERDICT: {follower} did really well!")
            self.speak(f"Excellent work, {follower}! You got most of it right. No drink for you - solid performance!")

        elif similarity_score >= 0.55:
            print(f"😅 MOMO'S VERDICT: {follower} captured the spirit!")
            self.speak(
                f"Not bad, {follower}! You got the main moves but missed some details. Take a small sip for the imperfections!")

        elif similarity_score >= 0.40:
            print(f"🤔 MOMO'S VERDICT: {follower} tried their best!")
            self.speak(f"Oh {follower}, you tried but there were significant differences. Take a drink for the effort!")

        else:
            print(f"🤖 MOMO'S VERDICT: Total algorithmic confusion!")
            self.speak(f"My AI is having an existential crisis! Both of you drink for breaking my neural networks!")

        # Detailed breakdown
        self.provide_detailed_feedback(leader, follower, similarity_score, leader_metrics, follower_metrics)

        print(f"\n🏆 SIPSYNC COMPLETE!")
        self.speak("What an amazing dance battle!")

    def _calculate_body_only_similarity(self, poses1, poses2):
        """Calculate similarity for body poses only"""
        if not poses1 or not poses2:
            return 0.0

        similarities = []
        min_length = min(len(poses1), len(poses2))

        for i in range(min_length):
            if poses1[i] and poses2[i]:
                similarity = self._compare_body_landmarks(poses1[i], poses2[i])
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_hand_only_similarity(self, hands1, hands2):
        """Calculate similarity for hand gestures only"""
        if not hands1 or not hands2:
            return 0.0

        similarities = []
        min_length = min(len(hands1), len(hands2))

        for i in range(min_length):
            if hands1[i] and hands2[i]:
                similarity = self._compare_hand_landmarks(hands1[i], hands2[i])
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _calculate_face_only_similarity(self, faces1, faces2):
        """Calculate similarity for facial expressions only"""
        if not faces1 or not faces2:
            return 0.0

        similarities = []
        min_length = min(len(faces1), len(faces2))

        for i in range(min_length):
            if faces1[i] and faces2[i]:
                similarity = self._compare_face_landmarks(faces1[i], faces2[i])
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    # === Game Selection and Flow ===
    def show_game_menu(self):
        """Display game options and get player choice"""
        intro_text = (
            "Hola! I'm Momo – your fabulously chaotic guide to games! "

        )
        print(f"\n🎉 {intro_text}")
        self.speak(intro_text, rate=185)

        print("\n📋 GAME OPTIONS:")
        print("1. Face-Off - Keep a straight face (simplified facial tracking)")
        print("2. Blink Challenge - Stare without blinking (face mesh tracking)")
        print("3. SipSync - Full body, hand, and face dance battle")
        print("4. Riddle Rush - Solve brain teasers in 10 seconds")

        while True:
            self.speak(
                "Choose your chaos! Type 1 for Face-Off, 2 for Blink Challenge, 3 for SipSync, or 4 for Riddle Rush.")
            choice = input("\n👉 Enter your choice (1, 2, 3, or 4): ").strip()

            if choice in ['1', '2', '3', '4']:
                return choice
            else:
                print("😵 Momo is confused. Pick 1, 2, 3, or 4!")
                self.speak("Pick a number between 1 and 4, human.")

    def explain_game(self, choice):
        """Explain the game rules"""
        explanations = {
            '1': "Face-Off Rules: This is a 3 rounds game - You need to keep a straight face while I blast you with jokes-  !",
            '2': "Blink Challenge Rules: Stare at the camera without blinking for 15 seconds",
            '3': "SipSync Rules: Ultimate 2-player dance battle!  Copy well = no drink, copy poorly = drink!",
            '4': "Riddle Rush Rules: I'll give you 2 riddles, 10 seconds each to answer. Get at least ONE correct and you're safe from drinking. Get ZERO correct and you drink!"
        }

        explanation = explanations[choice]
        print(f"\n📢 {explanation}")
        self.speak(explanation, rate=185)

    # === Main Game Loop ===
    def play_game(self):
        """Main game execution loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Couldn't access the webcam.")
            self.speak("I can't see you! Check your camera.")
            return

        try:
            print("🎮 MOMO'S DRINKING GAME STARTED!")
            self.speak("Welcome to Momo's drinking game extravaganza!")

            while True:
                # Game selection
                selected_game = self.show_game_menu()
                self.explain_game(selected_game)

                # Game execution
                if selected_game == '1':
                    self.play_face_off(cap)
                elif selected_game == '2':
                    self.play_blink_challenge(cap)
                elif selected_game == '3':
                    self.play_sipsync(cap)
                elif selected_game == '4':
                    self.play_riddle_rush(cap)

                # Continue playing?
                self.speak("Wanna play another round? My brain is getting smarter...", rate=180)
                answer = input("\n👉 Play another round? (y/n): ").strip().lower()

                if answer not in ['y', 'yes']:
                    break

        except KeyboardInterrupt:
            print("\n⚠️ Game interrupted!")

        finally:
            cap.release()
            cv2.destroyAllWindows()

        # Game over
        farewell = "Momo out! You were beautifully tracked in every detail, and I analyzed every pixel! Game over, you magnificent specimens!"
        print(f"\n👋 {farewell}")
        self.speak(farewell, rate=200)

    # === Additional Features ===
    def analyze_dance_quality(self, poses, hands, faces):
        """Analyze overall dance quality with detailed metrics"""
        metrics = {
            'body_movement_variety': 0,
            'hand_gesture_complexity': 0,
            'facial_expression_changes': 0,
            'overall_energy': 0
        }

        if not poses:
            return metrics

        # Analyze body movement variety
        if len(poses) > 1:
            movement_changes = 0
            for i in range(1, len(poses)):
                if poses[i] and poses[i - 1]:
                    change = self._calculate_pose_change(poses[i - 1], poses[i])
                    movement_changes += change
            metrics['body_movement_variety'] = movement_changes / len(poses)

        # Analyze hand gesture complexity
        if hands:
            hand_changes = 0
            valid_hand_frames = 0
            for i in range(1, len(hands)):
                if hands[i] and hands[i - 1]:
                    change = self._calculate_hand_change(hands[i - 1], hands[i])
                    hand_changes += change
                    valid_hand_frames += 1
            if valid_hand_frames > 0:
                metrics['hand_gesture_complexity'] = hand_changes / valid_hand_frames

        # Analyze facial expression changes
        if faces:
            face_changes = 0
            valid_face_frames = 0
            for i in range(1, len(faces)):
                if faces[i] and faces[i - 1]:
                    change = self._calculate_face_change(faces[i - 1], faces[i])
                    face_changes += change
                    valid_face_frames += 1
            if valid_face_frames > 0:
                metrics['facial_expression_changes'] = face_changes / valid_face_frames

        # Calculate overall energy
        metrics['overall_energy'] = (
                metrics['body_movement_variety'] * 0.5 +
                metrics['hand_gesture_complexity'] * 0.3 +
                metrics['facial_expression_changes'] * 0.2
        )

        return metrics

    def _calculate_pose_change(self, pose1, pose2):
        """Calculate amount of change between two poses"""
        total_change = 0
        valid_landmarks = 0

        for i in range(min(len(pose1.landmark), len(pose2.landmark))):
            lm1 = pose1.landmark[i]
            lm2 = pose2.landmark[i]

            if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                change = self.euclidean_distance_3d(lm1, lm2)
                total_change += change
                valid_landmarks += 1

        return total_change / valid_landmarks if valid_landmarks > 0 else 0

    def _calculate_hand_change(self, hands1, hands2):
        """Calculate amount of change between hand positions"""
        if not hands1 or not hands2:
            return 0

        changes = []
        for h1 in hands1:
            for h2 in hands2:
                total_change = 0
                for i in range(min(len(h1.landmark), len(h2.landmark))):
                    change = self.euclidean_distance_3d(h1.landmark[i], h2.landmark[i])
                    total_change += change
                changes.append(total_change / len(h1.landmark))

        return max(changes) if changes else 0

    def _calculate_face_change(self, face1, face2):
        """Calculate amount of change between facial expressions"""
        if not face1 or not face2:
            return 0

        # Focus on expression-critical points
        expression_points = [61, 291, 13, 14, 78, 308, 11, 12]  # Mouth and eyebrow points
        total_change = 0
        valid_points = 0

        for point_idx in expression_points:
            if (point_idx < len(face1.landmark) and point_idx < len(face2.landmark)):
                change = self.euclidean_distance(face1.landmark[point_idx], face2.landmark[point_idx])
                total_change += change
                valid_points += 1

        return total_change / valid_points if valid_points > 0 else 0

    def provide_detailed_feedback(self, leader_name, follower_name, similarity_score, leader_metrics, follower_metrics):
        """Provide detailed AI feedback on the dance battle"""
        print(f"\n🤖 AI DETAILED ANALYSIS:")
        print(f"=" * 50)

        print(f"\n👑 {leader_name}'s Performance:")
        print(f"   Body Movement Variety: {leader_metrics['body_movement_variety']:.3f}")
        print(f"   Hand Gesture Complexity: {leader_metrics['hand_gesture_complexity']:.3f}")
        print(f"   Facial Expression Changes: {leader_metrics['facial_expression_changes']:.3f}")
        print(f"   Overall Energy: {leader_metrics['overall_energy']:.3f}")

        print(f"\n🎯 {follower_name}'s Performance:")
        print(f"   Body Movement Variety: {follower_metrics['body_movement_variety']:.3f}")
        print(f"   Hand Gesture Complexity: {follower_metrics['hand_gesture_complexity']:.3f}")
        print(f"   Facial Expression Changes: {follower_metrics['facial_expression_changes']:.3f}")
        print(f"   Overall Energy: {follower_metrics['overall_energy']:.3f}")

        print(f"\n📊 Similarity Analysis:")
        print(f"   Overall Similarity: {similarity_score:.3f}")

        # ✅ BRUTAL SARCASTIC COMMENTARY BASED ON SIMILARITY SCORE
        print(f"\n💬 AI Commentary:")

        if similarity_score >= 0.85:
            # Excellent performance - praise them
            energy_comment = f"{leader_name} brought amazing energy!"
            copy_comment = f"{follower_name} nailed it perfectly!"
            brutal_comment = "Wow, actual talent detected! No drinks for you champions!"

        elif similarity_score >= 0.70:
            # Good performance - mild praise
            energy_comment = f"{leader_name} showed solid moves!"
            copy_comment = f"{follower_name} copied it really well!"
            brutal_comment = "Not bad at all! You actually tried and succeeded!"

        elif similarity_score >= 0.55:
            # Mediocre performance - gentle roasting
            energy_comment = f"{leader_name} had some decent moments."
            copy_comment = f"{follower_name} got the general idea but missed details."
            brutal_comment = "Eh, you tried. Take a small sip for the effort!"

        elif similarity_score >= 0.40:
            # Poor performance - moderate roasting
            energy_comment = f"{leader_name}'s performance was... interesting."
            copy_comment = f"{follower_name} interpreted it very creatively."
            brutal_comment = "Well that was... something. Drink up for that creative interpretation!"

        else:
            # Terrible performance - BRUTAL SARCASM
            sarcastic_comments = [
                "OH MY GOD! What did I just witness? Did you both have strokes mid-dance?",
                "I've seen mannequins with more rhythm! Both of you, drink immediately!",
                "That was so bad, my AI is considering retirement! DRINK!",
                "Were you dancing or having seizures? I honestly can't tell. DRINK UP!",
                "I asked for a dance battle, not a medical emergency! Both of you, DRINK!",
                "That similarity score is lower than my expectations, and they were already underground!",
                "Congratulations! You just broke the laws of physics AND choreography!"
            ]

            brutal_comment = random.choice(sarcastic_comments)
            energy_comment = f"{leader_name} brought chaos, not energy."
            copy_comment = f"{follower_name} copied absolutely nothing correctly."

        print(f"   🔥 MOMO'S BRUTAL TRUTH: {brutal_comment}")
        print(f"   📈 Performance: {energy_comment}")
        print(f"   🎯 Copying Skills: {copy_comment}")

        # Additional savage commentary for really bad scores
        if similarity_score < 0.40:
            savage_extras = [
                "   💀 I'm questioning my entire existence after watching that.",
                "   🤖 Error 404: Dance skills not found.",
                "   😵 That was painful to analyze. My neural networks are crying.",
                "   🆘 Should I call emergency services? That looked dangerous.",
                "   🎭 Next time, just stand still. It would be more accurate."
            ]
            print(random.choice(savage_extras))

        # Speak the brutal truth
        if similarity_score < 0.40:
            self.speak(
                f"Oh honey, that was TERRIBLE! {brutal_comment} Both of you need to drink and think about what you just did!")
        else:
            self.speak(f"Analysis complete! {brutal_comment}")

    # === Debug and Calibration Methods ===
    def test_tracking_systems(self, cap):
        """Test all tracking systems to ensure they're working"""
        print("\n🔧 Testing Tracking Systems...")
        self.speak("Testing all tracking systems. Move around, make gestures, and show expressions!")

        test_duration = 10
        start_time = time.time()

        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                continue

            # Test all tracking
            pose_landmarks, hand_landmarks, face_landmarks = self.extract_all_features(frame)

            # Display tracking status
            status_y = 30
            if pose_landmarks:
                cv2.putText(frame, "Body: TRACKING", (30, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.mp_drawing.draw_landmarks(frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            else:
                cv2.putText(frame, "Body: NOT DETECTED", (30, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            status_y += 30
            if hand_landmarks:
                cv2.putText(frame, f"Hands: {len(hand_landmarks)} DETECTED", (30, status_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                for hand in hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
            else:
                cv2.putText(frame, "Hands: NOT DETECTED", (30, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            status_y += 30
            if face_landmarks:
                cv2.putText(frame, "Face: TRACKING", (30, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.mp_drawing.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
            else:
                cv2.putText(frame, "Face: NOT DETECTED", (30, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show remaining time
            remaining = int(test_duration - (time.time() - start_time))
            cv2.putText(frame, f'Test Time: {remaining}s', (30, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)

            cv2.imshow("Tracking Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        cv2.destroyAllWindows()
        print("✅ Tracking system test complete!")
        self.speak("All tracking systems tested successfully!")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        print("🚀 Initializing Momo Game with Simplified Facial Expression Detection!")

        game = MomoGame()

        # Optional: Test tracking systems first
        test_mode = input("\n🔧 Run tracking system test first? (y/n): ").strip().lower()
        if test_mode in ['y', 'yes']:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                game.test_tracking_systems(cap)
                cap.release()
            else:
                print("❌ Camera not available for testing")

        # Start main game
        game.play_game()

    except KeyboardInterrupt:
        print("\n👋 Momo got bored. Thanks for playing!")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("📦 Install with: pip install deepface opencv-python mediapipe pyttsx3 scipy numpy")
    except Exception as e:
        print(f"💥 Something went wrong: {e}")
        print("Don't worry, Momo still loves you!")