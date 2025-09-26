"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const navLinks = [
  { name: 'How It Works', href: '#' },
  { name: 'Support', href: '#' },
  { name: 'About', href: '#' },
  { name: 'Blog', href: '#' },
];

export function Header() {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    const timeout = setTimeout(() => setIsMounted(true), 100);
    return () => clearTimeout(timeout);
  }, []);

  return (
    <header
      className={cn(
        'fixed top-4 left-1/2 -translate-x-1/2 z-50 w-[95%] max-w-6xl transition-all duration-700 ease-out',
        isMounted ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-8'
      )}
    >
      <div className="rounded-full px-4 py-2 backdrop-blur-lg bg-[linear-gradient(180deg,rgba(255,255,255,0.10)_0%,rgba(255,255,255,0.04)_100%)] border border-white/15 shadow-[0_8px_40px_rgba(0,0,0,0.35)]">
        <nav className="flex items-center justify-between gap-4">
          <Link href="/" className="text-xl font-bold tracking-wider font-sans">
            TRUTHLENS AI
          </Link>
          
          <ul className="hidden lg:flex items-center gap-x-6">
            {navLinks.map((link) => (
              <li key={link.name}>
                <Link href={link.href} className="text-sm text-foreground/80 hover:text-foreground transition-colors">
                  {link.name}
                </Link>
              </li>
            ))}
          </ul>

          <div className="flex items-center gap-x-2 sm:gap-x-4">
            <Link href="/login" className="hidden sm:block text-sm text-foreground/80 hover:text-foreground transition-colors">
              Log In
            </Link>
            <Link href="/analyze">
                <Button variant="outline" className="rounded-full bg-transparent hover:bg-white/10 border-white/50 h-9 px-4 sm:px-6 text-sm">
                Analyze Media
                </Button>
            </Link>
          </div>
        </nav>
      </div>
    </header>
  );
}