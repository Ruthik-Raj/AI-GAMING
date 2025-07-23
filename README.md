# 🎮 Momo's Drinking Game

Welcome to **Momo's Drinking Game** - an AI-powered party game that combines computer vision, facial recognition, pose detection, and sarcastic commentary for the ultimate interactive drinking game experience! 

This innovative project transforms your webcam into a sophisticated motion detection system, capable of analyzing micro-expressions, full-body movements, and even detecting when you blink. Built with cutting-edge machine learning libraries, Momo creates an immersive gaming experience where artificial intelligence meets social entertainment. Whether you're hosting a party, hanging out with friends, or just want to test your skills against an unforgiving AI judge, this game delivers laughs, challenges, and probably a few drinks along the way.

The game leverages real-time computer vision processing to create dynamic, responsive gameplay that adapts to each player's performance. Every gesture, expression, and movement is tracked with scientific precision, then evaluated by Momo's brutally honest AI personality system that provides instant feedback with a mix of encouragement and savage roasts.

## 🤖 Meet Momo

Momo is your fabulously chaotic AI game master who uses advanced computer vision to track your every move, expression, and gesture. She's got attitude, jokes, and zero mercy when judging your performance!

## 🎯 Game Modes

### 1. 🎭 Face-Off
- **Challenge**: Keep a straight face while Momo tells hilarious jokes
- **Technology**: DeepFace emotion detection
- **Rounds**: 3 rounds of comedic torture
- **Win Condition**: Don't crack up or you drink!

### 2. 👁️ Blink Challenge  
- **Challenge**: Stare at the camera without blinking for 15 seconds
- **Technology**: MediaPipe Face Mesh with Eye Aspect Ratio (EAR) calculation
- **Detection**: Advanced blink detection using facial landmarks
- **Win Condition**: Don't blink or everyone else drinks!

### 3. 🕺 SipSync - Ultimate Dance Battle
- **Challenge**: Two-player dance copying competition
- **Technology**: Full-body tracking with MediaPipe (33 pose landmarks + hands + face)
- **Features**: 
  - Real-time pose similarity analysis
  - Hand gesture tracking (21 landmarks per hand)
  - Facial expression matching
  - AI-powered performance scoring
- **Win Condition**: Copy the dance well = no drink, fail = drink up!

### 4. 🧩 Riddle Rush
- **Challenge**: Solve 2 riddles in 10 seconds each
- **Features**: 45+ riddles with smart answer matching
- **Win Condition**: Get at least 1 correct to avoid drinking

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV, MediaPipe
- **AI/ML**: DeepFace for emotion recognition
- **Audio**: pyttsx3 for text-to-speech
- **Mathematics**: SciPy for distance calculations
- **Core**: Python 3.7+

## 📋 Requirements

```bash
pip install deepface opencv-python mediapipe pyttsx3 scipy numpy
```

### System Requirements
- Python 3.7 or higher
- Webcam (built-in or external)
- Microphone/speakers for audio feedback
- Good lighting for optimal face/pose detection

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/momos-drinking-game.git
   cd momos-drinking-game
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the game**
   ```bash
   python momo_game.py
   ```

## 🎮 How to Play

1. **Launch the game** and ensure your webcam is working
2. **Choose your game mode** (1-4) when prompted
3. **Follow Momo's instructions** - she'll guide you through each challenge
4. **Get ready for brutal AI commentary** on your performance!

### Controls
- **ESC**: Exit any game mode
- **Camera positioning**: Make sure you're fully visible in the frame
- **Audio**: Momo will speak instructions and commentary

## 🔧 Features

### Advanced Computer Vision
- **33-point full body pose tracking**
- **21-point hand landmark detection per hand**
- **468-point facial mesh analysis**
- **Real-time similarity scoring algorithm**
- **Blink detection with Eye Aspect Ratio**

### AI Personality
- **Sarcastic commentary** based on performance
- **Dynamic difficulty adjustment**
- **Personalized feedback** with detailed metrics
- **45+ jokes and riddles** for variety

### Performance Analysis
- Body movement variety scoring
- Hand gesture complexity analysis
- Facial expression change detection
- Overall energy level calculation

## 🎨 Game Screenshots

*Add screenshots of your game in action here*

## 🧪 Testing Mode

Run the tracking system test to ensure all computer vision components are working:

```python
# When prompted at startup
"Run tracking system test first? (y/n): y"
```

This will show real-time tracking of:
- ✅ Body pose landmarks
- ✅ Hand detection and tracking  
- ✅ Facial mesh overlay
- ✅ System performance metrics

## 🤝 Contributing

We welcome contributions! Here are some ways you can help:

- **Add new game modes** with different CV challenges
- **Improve AI commentary** with more personality
- **Enhance pose similarity algorithms** for better accuracy
- **Add new riddles/jokes** to the content database
- **Optimize performance** for different hardware setups

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Check camera permissions and try different camera indices
cap = cv2.VideoCapture(1)  # Try 1, 2, etc.
```

**Poor tracking performance:**
- Ensure good lighting
- Position yourself 3-6 feet from camera
- Wear contrasting colors
- Remove background clutter

**Audio issues:**
- Check system audio settings
- Install additional TTS voices if needed
- Adjust speaking rate in code if too fast/slow

**Installation problems:**
```bash
# For Windows users with TTS issues:
pip install pypiwin32

# For Mac users:
brew install portaudio
pip install pyaudio
```

## 📊 Performance Metrics

The game tracks detailed performance metrics:

- **Similarity Scores**: 0.0-1.0 scale for pose matching
- **Reaction Times**: Millisecond precision for challenges  
- **Accuracy Rates**: Percentage-based scoring systems
- **Engagement Levels**: Based on movement variety and energy

## 🎉 Fun Facts

- Momo analyzes **33 body landmarks**, **42 hand points**, and **468 facial landmarks** in real-time
- The similarity algorithm processes over **500 coordinate points** per frame
- Blink detection uses mathematical eye aspect ratio calculations
- The AI commentary has multiple personality modes based on performance

## 🔮 Future Features

- [ ] Multiplayer support for larger parties
- [ ] Custom game mode creator
- [ ] Performance analytics dashboard  
- [ ] Mobile app version
- [ ] VR/AR integration
- [ ] Online leaderboards
- [ ] More AI personalities beyond Momo

---

**⚠️ Drink Responsibly**: This game is intended for entertainment purposes. Please drink responsibly and follow local laws regarding alcohol consumption.

**🎮 Ready to get roasted by an AI?** Launch the game and let Momo judge your every move!

---

*Made with ❤️ and a lot of computer vision magic*
