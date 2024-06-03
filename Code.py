import ultralytics
from ultralytics import YOLO
import cv2

# Load Model (replace with your trained model path)
model = YOLO('best.pt')

# Define a dictionary mapping unknown class to similar classes (modify as needed)
unknown_similarity = {
    0:  ['insect', 'Spider'],
    1:  ['bird', 'Parrot'],
    2:  ['insect', 'Scorpion'],  # Technically an arachnid, but might be confused with insect
    3:  ['Chelonioidea', 'Sea turtle or turtle'],
    4:  ['ungulate', 'Cow or Buffalo'],
    5:  ['mammal', 'Canine or Jackels'],
    6:  ['mammal', 'Moles or Shrews'],
    7:  ['reptile','Tortoise or Terrapin'],
    8:  ['mammal', 'Domestic cat or Puma'],
    9:  ['reptile','Snake'],
    10: ['fish','Sawfish'],
    11: ['ungulate', 'Dunkey or Asses'],
    12: ['bird','Pied currawong'],
    13: ['rodent','American pika'],
    14: ['bird','Sapsucker'],
    15: ['bird','Turkey'],
    16: ['bird','Penguine'],
    17: ['insect','Butterfly'],
    18: ['mammal', 'Lion'],  # Large cat
    19: ['mammal', 'Weasel or Badgers'],
    20: ['mammal','Coati'],
    21: ['mammal', 'Hippopotamus'],  # Large herbivore
    22: ['mammal','Seals'],
    23: ['bird', 'Grouse or Turkey'],
    24: ['mammal','Hog or Peccaries'],
    25: ['bird','Owl'],
    26: ['insect', 'Larva'],
    27: ['mammal', 'Marsupial'],
    28: ['mammal', 'Bear'],  # Subtype of bear
    29: ['invertebrate', 'Squid'],
    30: ['mammal', 'Dolphins or Porpoise'],
    31: ['mammal', 'Seal'],
    32: ['bird','crow'],
    33: ['mammal','Rodent'],
    34: ['mammal', 'Tiger'],  # Large cat
    35: ['reptile','Lizard'],
    36: ['insect', 'Beetle'],
    37: ['mammal','Panda'],
    38: ['mammal', 'Kangaroo'],
    39: ['echinoderm','Fish'],
    40: ['invertebrate','Millipede'],
    41: ['reptile', 'Turtle'],
    42: ['bird', 'Rheas'],
    43: ['fish','Fish'],
    44: ['amphibian','Frog'],
    45: ['bird','Goose or Pelican'],
    46: ['mammal','Elephant'],
    47: ['mammal','Alpaca'],
    48: ['invertebrate', 'Slug'],
    49: ['mammal', 'Zebra or Okapi'],  # Similar to horse
    50: ['insect','Moth or Butterflies'],
    51: ['invertebrate', 'Prawn'],
    52: ['vertebrate', 'Fish'],
    53: ['mammal', 'Bear'],  # Subtype of bear
    54: ['mammal', 'Bob cat'],  # Similar to cat
    55: ['bird', 'Goose'],
    56: ['mammal', 'Ocelot'],  # Large cat
    57: ['bird', 'Duck'],
    58: ['mammal','Yalk or Bison'],
    59: ['mammal', 'Rodent'],
    60: ['mammal','Giraffe'],
    61: ['invertebrate', 'Crab'],
    62: ['invertebrate', 'Scorpions'],
    63: ['mammal', 'Lemurs or Apes'],
    64: ['mammal', 'Cattle'],
    65: ['fish','Seahorse'],
    66: ['invertebrate', 'Millipede'],
    67: ['mammal', 'Donkey'],  # Similar to horse
    68: ['mammal','Rhinoceros'],
    69: ['bird','Wild Canary'],
    70: ['mammal','Camel'],
    71: ['mammal', 'Bear'],  # Subtype of bear
    72: ['bird','Sparrow'],
    73: ['mammal', 'Rodent or Squirrel'],
    74: ['mammal', 'Leopard'],  # Large cat
    75: ['cnidarian','Fish'],
    76: ['reptile','Crocodiles'],
    77: ['mammal','Sambar'],
    78: ['bird', 'Turkey'],
    79: ['mammal', 'Seal'],
    80: ['human','Person'],
    81: ['vehicle', 'truck'],
    82: ['animal', 'mammal', 'canine'],
    83: ['animal', 'mammal', 'feline'],
    84: ['animal', 'flying creature'],  # Can be further narrowed down based on specific bird types
    85: ['arthropod', 'bug'],  # Can be further categorized (e.g., 'beetle', 'fly')
    86: ['aquatic animal', 'sea creature'],  # Can be further categorized (e.g., 'shark', 'ray')
    87: ['animal', 'cold-blooded'],  # Can be further categorized (e.g., 'snake', 'lizard')
    88: ['animal', 'mammal'],  # Can be further categorized (e.g., 'mouse', 'rat')
    89: ['animal', 'cold-blooded'],  # Can be further categorized (e.g., 'frog', 'toad')
    90: ['animal', 'mammal'],  # Can be further categorized (e.g., 'horse', 'cow')
    
}


# Class Names (replace with your actual class names)
class_names = ['Spider', 'Parrot', 'Scorpion', 'Sea turtle', 'Cattle', 'Fox', 'Hedgehog',
              'Turtle', 'Cheetah', 'Snake', 'Shark', 'Horse', 'Magpie', 'Hamster',
              'Woodpecker', 'Eagle', 'Penguin', 'Butterfly', 'Lion', 'Otter', 'Raccoon',
              'Hippopotamus', 'Bear', 'Chicken', 'Pig', 'Owl', 'Caterpillar', 'Koala',
              'Polar bear', 'Squid', 'Whale', 'Harbor seal', 'Raven', 'Mouse', 'Tiger',
              'Lizard', 'Ladybug', 'Red panda', 'Kangaroo', 'Starfish', 'Worm', 'Tortoise',
              'Ostrich', 'Goldfish', 'Frog', 'Swan', 'Elephant', 'Sheep', 'Snail', 'Zebra', 'Moth and butterflies', 
              'Shrimp', 'Fish', 'Panda', 'Lynx', 'Duck', 'Jaguar', 'Goose', 'Goat', 'Rabbit', 'Giraffe', 'Crab',
              'Tick', 'Monkey', 'Bull', 'Seahorse', 'Centipide', 'Mule', 'Rhinoceros', 'Canary', 'Camel', 'Brown Bera',
              'Sparrow', 'Squirrel', 'Leopard', 'Jellyfish', 'Crocodile', 'Deer','Turkey', 'Sea Lion',]

# Open Video Stream (or use '0' for webcam)
cap = cv2.VideoCapture("animal.mp4")

while cap.isOpened():
  success, frame = cap.read()

  if success:
    # Run Inference on Frame
    results = model.predict(frame,show=True)  # Assuming model takes OpenCV image directly

    for result in results:
      boxes = result.boxes

      for box in boxes:
        cls = (box.cls)  # Get predicted class
        conf = box.conf  # Get confidence score

        if cls.item() not in class_names:  # Check if class is unknown
          
          if  conf < 0.8:  # Check for high confidence
            # Suggest similar class based on the dictionary
            if cls.item()  in unknown_similarity :
              similar_class = unknown_similarity[cls.item()][0]
              print(f"Potential unknown species detected! Confidence: {conf.item():.2f}, Most similar species like: {similar_class}")
              similar_type = unknown_similarity[cls.item()][1]
              print(f"{conf.item():.2f}, Most similar animal like: {similar_type}")
            else:
              print(f"Potential unknown class detected! Confidence: {conf.item():.2f}, Class: {cls.item()}")

            # Draw Bounding Box and Label on Frame
            x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates from box object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green rectangle
            
            # Label text
            if cls.item() in unknown_similarity:
              similar_class = unknown_similarity[cls.item()][0]
              label_text = f"Unknown ({conf.item():.2f}) - Most similar to: {similar_class}"
              similar_type = unknown_similarity[cls.item()][1]
              label_text=f"Similar Species like ({conf.item():.2f}) - Most similar to: {similar_type}"
            else:
              label_text = f"{cls.item()} ({conf.item():.2f}) - {cls}"
            cv2.putText(frame, label_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    # Display Frame
    cv2.imshow('Unknown Species Detection:', frame)

    # Exit on 'q' Key Press
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# Release Video Capture and Close Windows
cap.release()
cv2.destroyAllWindows()