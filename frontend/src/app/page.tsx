import { Header } from "@/components/layout/header";
import { Hero } from "@/components/landing/hero";
import { CrisisSection } from "@/components/landing/crisis-section";
import { HowItWorksSection } from "@/components/landing/how-it-works-section";
import { FeatureShowcaseSection } from "@/components/landing/feature-showcase-section";
import { CtaSection } from "@/components/landing/cta-section";
import { Footer } from "@/components/layout/footer";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Header />
      <main className="flex-1">
        <Hero />
        <CrisisSection />
        <div className="h-px w-full bg-gradient-to-r from-transparent via-foreground/10 to-transparent" />
        <HowItWorksSection />
        <div className="h-px w-full bg-gradient-to-r from-transparent via-foreground/10 to-transparent" />
        <FeatureShowcaseSection />
        <div className="h-px w-full bg-gradient-to-r from-transparent via-foreground/10 to-transparent" />
        <CtaSection />
      </main>
      <Footer />
    </div>
  );
}
