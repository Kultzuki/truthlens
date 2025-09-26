"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Landmark, ShieldAlert, Users } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

const crisisPoints = [
  {
    icon: <Landmark className="w-10 h-10 text-primary" />,
    title: "Election Manipulation",
    text: "Deepfakes can be used to create convincing but false videos of political candidates, spreading disinformation and influencing public opinion to sow social and political discord."
  },
  {
    icon: <ShieldAlert className="w-10 h-10 text-primary" />,
    title: "Online Scams & Frauds",
    text: "By impersonating trusted individuals or creating fake celebrity endorsements, malicious actors use deepfakes to execute sophisticated financial scams and steal sensitive information."
  },
  {
    icon: <Users className="w-10 h-10 text-primary" />,
    title: "Eroding Public Trust",
    text: "The proliferation of deepfakes threatens to undermine our collective trust in media, news outlets, and public institutions, making it harder to distinguish fact from fiction."
  }
];

export function CrisisSection() {
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
    <section ref={sectionRef} className="relative bg-background py-24 sm:py-32 overflow-hidden">
      {/* Animated Neon Grid Background */}
      <div className="absolute inset-0 -z-10">
        <svg aria-hidden="true" className="pointer-events-none absolute inset-0 text-primary/20 [mask-image:linear-gradient(to_bottom,transparent,black_10%,black_90%,transparent)]" width="100%" height="100%">
          <defs>
            <pattern id="grid-crisis" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1" className="animate-pulse"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid-crisis)" />
        </svg>
      </div>

      {/* Floating Neon Orbs */}
      <div style={parallax(0.01)} className="absolute inset-0 -z-10">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-primary/20 animate-pulse"
            style={{
              width: `${20 + Math.random() * 40}px`,
              height: `${20 + Math.random() * 40}px`,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${i * 0.5}s`,
              animationDuration: `${3 + Math.random() * 2}s`,
              boxShadow: `0 0 ${20 + Math.random() * 30}px hsl(var(--primary)/0.3)`,
            }}
          />
        ))}
      </div>

      {/* Animated Wave Separator */}
      <svg aria-hidden="true" viewBox="0 0 1440 120" className="absolute top-0 left-0 right-0 -z-10 text-primary/30 animate-pulse">
        <path fill="currentColor" d="M0,96L80,80C160,64,320,32,480,21.3C640,11,800,21,960,37.3C1120,53,1280,75,1360,85.3L1440,96L1440,0L1360,0C1280,0,1120,0,960,0C800,0,640,0,480,0C320,0,160,0,80,0L0,0Z"/>
      </svg>

      {/* Neon Laser Lines */}
      <div className="absolute inset-0 -z-10">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="absolute h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent animate-pulse"
            style={{
              top: `${20 + i * 15}%`,
              left: '0%',
              right: '0%',
              animationDelay: `${i * 0.3}s`,
              animationDuration: '4s',
              boxShadow: '0 0 10px hsl(var(--primary)/0.5)',
            }}
          />
        ))}
      </div>

      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className={cn(
          "mx-auto max-w-2xl lg:text-center transition-all duration-1000",
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        )}>
          <p className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl text-glow">
            Why It Matters
          </p>
          <p className="mt-6 text-lg leading-8 text-muted-foreground">
            In an age of digital information, the line between reality and fabrication is blurring. Understanding the risks is the first step toward protection.
          </p>
        </div>
        
        <div className={cn(
          "mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none transition-all duration-1000 delay-300",
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        )}>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
            {crisisPoints.map((point, index) => (
              <Card 
                key={point.title} 
                className={cn(
                  "relative glassmorphic hover:border-primary/50 transition-all duration-500 group overflow-hidden",
                  "hover:shadow-[0_0_30px_hsl(var(--primary)/0.3)] hover:scale-105",
                  isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8',
                  `delay-${index * 200}ms`
                )}
              >
                {/* Neon Border Animation */}
                <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary/20 via-primary/40 to-primary/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-sm" />
                
                {/* Animated Background Glow */}
                <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                

                <CardHeader className="relative z-10">
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      {point.icon}
                      <div className="absolute inset-0 bg-primary/20 rounded-full animate-pulse blur-sm" />
                    </div>
                    <CardTitle className="text-xl text-glow">{point.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="relative z-10">
                  <p className="text-base text-muted-foreground">{point.text}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Neon Gradient */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 -z-10 bg-gradient-to-b from-transparent via-primary/10 to-background/95" />
    </section>
  );
}