import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export function CtaSection() {
  return (
    <section className="relative bg-background/50 py-24 sm:py-32 overflow-hidden">
      {/* radial grain texture */}
      <svg aria-hidden="true" className="absolute inset-0 -z-10 opacity-50" width="100%" height="100%">
        <defs>
          <radialGradient id="cta-gradient" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="hsl(var(--primary)/0.12)" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>
          <filter id="noise" x="0" y="0">
            <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="2" stitchTiles="stitch"/>
            <feColorMatrix type="saturate" values="0"/>
            <feBlend mode="multiply" />
          </filter>
        </defs>
        <rect width="100%" height="100%" fill="url(#cta-gradient)" filter="url(#noise)" />
      </svg>

      {/* existing orb */}
      <div className="absolute inset-0 z-0 flex items-center justify-center opacity-70">
        <div className="absolute w-[560px] h-[560px] bg-primary/25 rounded-full blur-3xl animate-[orb-pulse_10s_ease-in-out_infinite]" />
      </div>
      <div className="relative z-10 mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl text-glow">
            Ready to Verify the Truth?
          </h2>
          <p className="mt-6 text-lg leading-8 text-muted-foreground">
            Try out our app now and become a part of the solution.
          </p>
          <div className="mt-10">
            <Link href="/analyze">
              <Button size="lg" className={cn(
                "rounded-full px-10 py-6 text-lg font-semibold",
                "bg-primary text-primary-foreground",
                "hover:bg-primary/90 transition-transform hover:scale-105",
                "pulse-ripple"
              )}>
                Start Your Analysis
              </Button>
            </Link>
          </div>
        </div>
      </div>
      {/* angled bottom separator */}
      <svg aria-hidden="true" viewBox="0 0 1440 120" className="absolute bottom-0 left-0 right-0 -z-10 text-primary/25">
        <path fill="currentColor" d="M0,64 L120,80 C240,96 480,128 720,112 C960,96 1200,32 1320,16 L1440,0 L1440,120 L0,120 Z" />
      </svg>
    </section>
  );
}
