"use client";

import Image from "next/image";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Zap, Layers, BrainCircuit, Smartphone } from "lucide-react";
import { PlaceHolderImages } from "@/lib/placeholder-images";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

const features = [
  {
    icon: <Zap className="w-8 h-8 text-primary" />,
    title: "Real-Time Analysis",
    description: "Get results in seconds, not minutes. Our system is optimized for speed and accuracy.",
    imageId: "feature-realtime"
  },
  {
    icon: <Layers className="w-8 h-8 text-primary" />,
    title: "Explainability",
    description: "We don't just give you a verdict, we show you why. Our heatmaps highlight the manipulated regions.",
    imageId: "feature-explainability"
  },
  {
    icon: <BrainCircuit className="w-8 h-8 text-primary" />,
    title: "Ensemble Learning",
    description: "Our system uses a combination of multiple AI models for higher accuracy and resilience against new deepfake techniques.",
    imageId: "feature-ensemble"
  },
  {
    icon: <Smartphone className="w-8 h-8 text-primary" />,
    title: "Mobile-Friendly UI",
    description: "Access our powerful tool from any device with a clean and intuitive design.",
    imageId: "feature-mobile"
  }
];

export function FeatureShowcaseSection() {
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

  const getImage = (id: string) => PlaceHolderImages.find(img => img.id === id);

  return (
    <section ref={sectionRef} className="relative bg-background py-24 sm:py-32 overflow-hidden">
      {/* Animated Neon Stripes Background */}
      <div className="absolute inset-0 -z-10">
        <svg aria-hidden="true" className="pointer-events-none absolute inset-0 text-primary/25" width="100%" height="100%">
          <defs>
            <pattern id="stripes" width="16" height="16" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="6" height="16" fill="currentColor" className="animate-pulse" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#stripes)" />
        </svg>
      </div>

      {/* Floating Neon Orbs */}
      <div style={parallax(0.006)} className="absolute inset-0 -z-10">
        {Array.from({ length: 15 }).map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-primary/12 animate-pulse"
            style={{
              width: `${10 + Math.random() * 20}px`,
              height: `${10 + Math.random() * 20}px`,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${i * 0.2}s`,
              animationDuration: `${3 + Math.random() * 2}s`,
              boxShadow: `0 0 ${10 + Math.random() * 20}px hsl(var(--primary)/0.5)`,
            }}
          />
        ))}
      </div>

      {/* Animated Laser Grid */}
      <div className="absolute inset-0 -z-10">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="absolute inset-0">
            <div
              className="absolute bg-gradient-to-r from-transparent via-primary/20 to-transparent animate-pulse"
              style={{
                width: '100%',
                height: '1px',
                top: `${i * 16.66}%`,
                animationDelay: `${i * 0.5}s`,
                animationDuration: '4s',
                boxShadow: '0 0 5px hsl(var(--primary)/0.4)',
              }}
            />
            <div
              className="absolute bg-gradient-to-b from-transparent via-primary/20 to-transparent animate-pulse"
              style={{
                width: '1px',
                height: '100%',
                left: `${i * 16.66}%`,
                animationDelay: `${i * 0.5}s`,
                animationDuration: '4s',
                boxShadow: '0 0 5px hsl(var(--primary)/0.4)',
              }}
            />
          </div>
        ))}
      </div>

      {/* Top Neon Divider */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent -z-10 animate-pulse" />

      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className={cn(
          "mx-auto max-w-2xl lg:text-center transition-all duration-1000",
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        )}>
          <p className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl text-glow">
            Why Choose Our App?
          </p>
          <p className="mt-6 text-lg leading-8 text-muted-foreground">
            Our advanced features provide unparalleled accuracy and insight, setting a new standard in deepfake detection.
          </p>
        </div>
        
        <div className={cn(
          "mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none transition-all duration-1000 delay-300",
          isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        )}>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
            {features.map((feature, index) => {
              const image = getImage(feature.imageId);
              return (
                <Card 
                  key={feature.title} 
                  className={cn(
                    "relative glassmorphic hover:border-primary/50 transition-all duration-700 flex flex-col group overflow-hidden",
                    "hover:shadow-[0_0_40px_hsl(var(--primary)/0.4)] hover:scale-[1.02]",
                    isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8',
                    `delay-${index * 200}ms`
                  )}
                >
                  {/* Professional Border Glow */}
                  <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary/10 via-primary/20 to-primary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  


                  <CardHeader className="relative z-10">
                    <div className="relative h-48 w-full mb-4 rounded-lg overflow-hidden group-hover:scale-105 transition-transform duration-700">
                      {image && (
                        <Image
                          src={image.imageUrl}
                          alt={image.description}
                          fill
                          className="object-cover"
                          data-ai-hint={image.imageHint}
                        />
                      )}
                      {/* Overlay Glow */}
                      <div className="absolute inset-0 bg-gradient-to-t from-primary/20 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="relative">
                        {feature.icon}
                        <div className="absolute inset-0 bg-primary/30 rounded animate-pulse blur-sm" />
                      </div>
                      <CardTitle className="text-xl text-glow group-hover:text-primary transition-colors duration-300">
                        {feature.title}
                      </CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent className="flex-grow relative z-10">
                    <p className="text-base text-muted-foreground group-hover:text-foreground/80 transition-colors duration-300">
                      {feature.description}
                    </p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </div>

      {/* Bottom Neon Divider */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent -z-10 animate-pulse" />
    </section>
  );
}