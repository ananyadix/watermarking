const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const authMiddleware = require("../middleware/authMiddleware");

const Patient = require("../models/Patient");
const Doctor = require("../models/Doctor");

const router = express.Router();


// ================= SIGNUP =================
router.post("/signup", async (req, res) => {
    try {
        const {
            name,
            email,
            password,
            role,
            phone,
            specialization,
            qualification,
            licenseNumber,
            experience
        } = req.body;

        // Check required fields
        if (!name || !email || !password || !role) {
            return res.status(400).json({ message: "Please fill all required fields" });
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        if (role === "patient") {

            const existingPatient = await Patient.findOne({ email });
            if (existingPatient) {
                return res.status(400).json({ message: "Patient already exists" });
            }

            await Patient.create({
                name,
                email,
                password: hashedPassword
            });

        } else if (role === "doctor") {

            const existingDoctor = await Doctor.findOne({ email });
            if (existingDoctor) {
                return res.status(400).json({ message: "Doctor already exists" });
            }

            // Optional: Check doctor extra fields
            if (!phone || !specialization || !qualification || !licenseNumber || !experience) {
                return res.status(400).json({ message: "Please fill all doctor details" });
            }

            await Doctor.create({
                name,
                email,
                password: hashedPassword,
                phone,
                specialization,
                qualification,
                licenseNumber,
                experience
            });

        } else {
            return res.status(400).json({ message: "Invalid role" });
        }

        res.status(201).json({ message: "Registered successfully" });

    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});


// ================= LOGIN =================
router.post("/login", async (req, res) => {
    try {
        const { email, password, role } = req.body;

        if (!email || !password || !role) {
            return res.status(400).json({ message: "Please provide email, password and role" });
        }

        let user;

        if (role === "patient") {
            user = await Patient.findOne({ email });
        } else if (role === "doctor") {
            user = await Doctor.findOne({ email });
        } else {
            return res.status(400).json({ message: "Invalid role" });
        }

        if (!user) {
            return res.status(400).json({ message: "User not found" });
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(400).json({ message: "Invalid credentials" });
        }

        const token = jwt.sign(
            { id: user._id, role },
            process.env.JWT_SECRET,
            { expiresIn: "1d" }
        );

        res.status(200).json({
            message: "Login successful",
            token,
            user: {
                id: user._id,
                name: user.name,
                email: user.email,
                role
            }
        });

    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});

// ================= ACCESSING DETAILS =================

router.get("/me", authMiddleware, async (req, res) => {
    try {
        let user;

        if (req.user.role === "patient") {
            user = await Patient.findById(req.user.id).select("-password");
        } else if (req.user.role === "doctor") {
            user = await Doctor.findById(req.user.id).select("-password");
        }

        res.json(user);

    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});

module.exports = router;
