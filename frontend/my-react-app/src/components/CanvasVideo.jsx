import React, { useRef, useEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const frameCount = 1000;

const currentFrame = index => (
  `/frames/frame_${(index + 1).toString().padStart(4, '0')}.jpg`
);

export default function CanvasVideo() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    // Set canvas dimensions (matches the extracted video 1080p)
    canvas.width = 1920;
    canvas.height = 1080;

    const images = [];
    const imageSeq = {
      frame: 0
    };

    // Preload all frames
    for (let i = 0; i < frameCount; i++) {
      const img = new Image();
      img.src = currentFrame(i);
      images.push(img);
    }

    // Draw first frame once loaded
    images[0].onload = () => {
      context.drawImage(images[0], 0, 0);
    };

    gsap.to(imageSeq, {
      frame: frameCount - 1, // Play 100% of the video
      snap: "frame",
      ease: "none",
      scrollTrigger: {
        trigger: ".app-container",
        start: "top top",
        end: "bottom bottom",
        scrub: 0.5 // Faster, more responsive scrub
      },
      onUpdate: () => {
        // Draw the current frame to the canvas
        if (images[imageSeq.frame]) {
          context.drawImage(images[imageSeq.frame], 0, 0);
        }
      }
    });

    return () => {
      ScrollTrigger.getAll().forEach(st => st.kill());
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="background-video"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        objectFit: 'cover',
        zIndex: -1,
        opacity: 1
      }}
    />
  );
}
