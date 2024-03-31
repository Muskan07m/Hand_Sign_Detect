document.addEventListener("DOMContentLoaded", function() {
    console.log("DOMContentLoaded event triggered");

    const tryDemoButton = document.getElementById("try-demo-btn");
    const howItWorksLink = document.getElementById("how-it-works-link");
    const howItWorksSection = document.querySelector(".how-it-works");
    const FeatureLink = document.getElementById("Features-link");
    const FaeturesSection = document.querySelector(".Features");
    const aboutLink = document.getElementById("about-us-link");
    const aboutSection = document.querySelector(".about-us");

    tryDemoButton.addEventListener("click", async function() {
        console.log("Try Demo button clicked");

        // Execute the try_demo endpoint to run test.py
        try {
            const response = await fetch('/try_demo', { method: 'POST' });
            const data = await response.json();
            console.log("Response:", data);
        } catch (error) {
            console.error("Error running try_demo:", error);
        }
    });

    howItWorksLink.addEventListener("click", function(event) {
        console.log("How It Works link clicked");

        event.preventDefault(); // Prevent the default action of the link

        // Toggle the display of the "How It Works" section
        if (howItWorksSection.style.display === "none" || howItWorksSection.style.display === "") {
            howItWorksSection.style.display = "block"; // Show the section
        } else {
            howItWorksSection.style.display = "none"; // Hide the section
        }
    });

    FeatureLink.addEventListener("click", function(event) {
        console.log("Features link clicked");

        event.preventDefault(); // Prevent the default action of the link

        // Toggle the display of the "How It Works" section
        if (FaeturesSection.style.display === "none" || FaeturesSection.style.display === "") {
            FaeturesSection.style.display = "block"; // Show the section
        } else {
            FaeturesSection.style.display = "none"; // Hide the section
        }
    });

    aboutLink.addEventListener("click", function(event) {
        console.log("About-Us link clicked");

        event.preventDefault(); // Prevent the default action of the link

        // Toggle the display of the "How It Works" section
        if (aboutSection.style.display === "none" || aboutSection.style.display === "") {
            aboutSection.style.display = "block"; // Show the section
        } else {
            aboutSection.style.display = "none"; // Hide the section
        }
    });


});