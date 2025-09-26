"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { UploadCloud, BrainCircuit, FileCheck } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

const steps = [
  {
    icon: <UploadCloud className="w-12 h-12 text-primary" />,
    title: "Step 1: Upload",
    text: "Easily upload any suspicious video, image, or audio file from your device or paste a URL."
  },
  {
    icon: <BrainCircuit className="w-12 h-12 text-primary" />,
    title: "Step 2: Analyze",
    text: "Our AI runs a real-time analysis, scanning for manipulation markers invisible to the naked eye."
  },
  {
    icon: <FileCheck className="w-12 h-12 text-primary" />,
    title: "Step 3: Verdict",
    text: "Receive a clear verdict with a confidence score and a visual heatmap showing manipulated areas."
  }
];

export function HowItWorksSection() {
  const [isVisible, setIsVisible] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    const handleMouseMove = (event: MouseEvent) => {
      setMousePosition({ x: event.clientX, y: event.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      observer.disconnect();
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const parallax = (speed: number) => {
    if (typeof window === 'undefined') return {};
    const x = (mousePosition.x - window.innerWidth / 2) * speed;
    const y = (mousePosition.y - window.innerHeight / 2) * speed;
    return { transform: `translate3d(${x}px, ${y}px, 0)` };
  };

  return (
    <section ref={sectionRef} className="relative bg-background/80 py-24 sm:py-32 overflow-hidden">
      {/* Cyberpunk Diagonal Grid */}
      <div className="absolute inset-0 -z-10">
        <svg aria-hidden="true" className="pointer-events-none absolute inset-0 text-primary/15" width="100%" height="100%">
          <defs>
            <pattern id="cyber-grid" width="60" height="60" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <path d="M 60 0 L 0 0 0 60" fill="none" stroke="currentColor" strokeWidth="1" className="animate-pulse"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#cyber-grid)" />
        </svg>
      </div>

      {/* Floating Neon Orbs */}
      <div style={parallax(0.008)} className="absolute inset-0 -z-10">
        {Array.from({ length: 12 }).map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-primary/15 animate-pulse"
            style={{
              width: `${15 + Math.random() * 25}px`,
              height: `${15 + Math.random() * 25}px`,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${i * 0.3}s`,
              animationDuration: `${4 + Math.random() * 3}s`,
              boxShadow: `0 0 ${15 + Math.random() * 25}px hsl(var(--primary)/0.4)`,
            }}
          />
        ))}
      </div>

      {/* Animated Circuit Lines */}
      <div className="absolute inset-0 -z-10">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="absolute bg-gradient-to-r from-transparent via-primary/30 to-transparent animate-pulse"
            style={{
              width: `${200 + Math.random() * 300}px`,
              height: '1px',
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              transform: `rotate(${Math.random() * 360}deg)`,
              animationDelay: `${i * 0.4}s`,
              animationDuration: '3s',
              boxShadow: '0 0 8px hsl(var(--primary)/0.6)',
            }}
          />
        ))}
      </div>

      {/* Animated Dotted Wave */}
      <svg aria-hidden="true" className="absolute -z-10 inset-x-0 top-0 opacity-60" viewBox="0 0 1440 120" preserveAspectRatio="none">
        <defs>
          <pattern id="dots-hiw" width="12" height="12" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="1" className="fill-primary/40 animate-pulse" />
          </pattern>
        </defs>
        <path d="M0,80 C200,20 400,20 600,80 C800,140 1000,140 1200,80 C1300,50 1400,50 1440,80 L1440,0 L0,0 Z" fill="url(#dots-hiw)" />
      </svg>

      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className={cn(
          "mx-auto max-w-2xl lg:text-center transition-all duration-1000",
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        )}>
          <p className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl text-glow">
            How We Fight Back
          </p>
          <p className="mt-6 text-lg leading-8 text-muted-foreground">
            Our three-step process makes sophisticated deepfake detection simple and accessible for everyone.
          </p>
        </div>
        
        <div className={cn(
          "mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none transition-all duration-1000 delay-300",
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        )}>
          <div className="grid grid-cols-1 gap-10 md:grid-cols-3">
            {steps.map((step, index) => (
              <div 
                key={step.title} 
                className={cn(
                  "relative flex flex-col items-center text-center group transition-all duration-700",
                  isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8',
                  `delay-${index * 300}ms`
                )}
              >
                {/* Main Icon Container with Halo */}
                <div className="relative flex items-center justify-center h-24 w-24 mb-6">
                  {/* Animated Neon Halo */}
                  <div className="absolute inset-0 -m-6 rounded-full bg-gradient-radial from-primary/30 via-primary/10 to-transparent animate-pulse blur-sm" />
                  <div className="absolute inset-0 -m-6 rounded-full border border-primary/20 animate-spin" style={{ animationDuration: '20s' }} />
                  
                  {/* Icon Container */}
                  <div className="relative flex items-center justify-center h-24 w-24 rounded-full bg-gradient-to-br from-primary/15 to-primary/5 ring-2 ring-primary/30 group-hover:ring-primary/60 group-hover:bg-gradient-to-br group-hover:from-primary/20 group-hover:to-primary/10 group-hover:scale-105 transition-all duration-300 shadow-lg group-hover:shadow-xl">
                    {step.icon}
                  </div>
                </div>

                <h3 className="text-xl font-semibold text-foreground text-glow group-hover:text-primary transition-colors duration-300">
                  {step.title}
                </h3>
                <p className="mt-2 text-base text-muted-foreground group-hover:text-foreground/80 transition-colors duration-300">
                  {step.text}
                </p>

              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Animated Bottom Curve */}
      <svg aria-hidden="true" viewBox="0 0 1440 100" className="absolute bottom-0 left-0 right-0 -z-10 text-primary/20 rotate-180">
        <path fill="currentColor" d="M0,64L60,58.7C120,53,240,43,360,53.3C480,64,600,96,720,101.3C840,107,960,85,1080,69.3C1200,53,1320,43,1380,37.3L1440,32L1440,0L1380,0C1320,0,1200,0,1080,0C960,0,840,0,720,0C600,0,480,0,360,0C240,0,120,0,60,0L0,0Z" />
      </svg>
    </section>
  );
}