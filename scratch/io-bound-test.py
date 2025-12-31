VIDEO = "../data/raw/5-test_folder/video/3-24.mp4"
import cv2, time, os, subprocess

def full_read():
    cap = cv2.VideoCapture(VIDEO)
    count = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    t1 = time.perf_counter()
    cap.release()
    return count, t1 - t0

def quick_probe(n=300):
    cap = cv2.VideoCapture(VIDEO)
    t0 = time.perf_counter()
    for _ in range(n):
        ret, f = cap.read()
        if not ret:
            break
    t1 = time.perf_counter()
    cap.release()
    return (t1 - t0) / max(1, n)

def quick_grab(n=300):
    cap = cv2.VideoCapture(VIDEO)
    t0 = time.perf_counter()
    for _ in range(n):
        cap.grab()
    t1 = time.perf_counter()
    cap.release()
    return (t1 - t0) / n

print("Frame shape & first 5 reads:")
cap = cv2.VideoCapture(VIDEO)
for i in range(5):
    ret, frame = cap.read()
    print(i, ret, None if frame is None else frame.shape)
cap.release()

count, tot = full_read()
print("Full read:", count, "frames in", tot, "s => avg ms/frame", tot/count*1000 if count else None)

print("Quick read avg ms:", quick_probe()*1000)
print("Quick grab avg ms:", quick_grab()*1000)

print("\nFFmpeg decode baseline (this may take a moment):")
subprocess.run(["ffmpeg", "-i", VIDEO, "-f", "null", "-"], stderr=subprocess.STDOUT)
