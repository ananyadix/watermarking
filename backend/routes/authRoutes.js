const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../models/User");
const authMiddleware = require("../middleware/authMiddleware");
const roleMiddleware = require("../middleware/roleMiddleware");

const router = express.Router();


// ================= SIGNUP =================
router.post("/signup", async (req, res) => {
    try {
        const { name, email, password, role } = req.body;

        const existingUser = await User.findOne({ email });
        if (existingUser)
            return res.status(400).json({ message: "User already exists" });

        const hashedPassword = await bcrypt.hash(password, 10);

        const user = await User.create({
            name,
            email,
            password: hashedPassword,
            role
        });

        res.status(201).json({ message: "User registered successfully" });

    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});


// ================= LOGIN =================
router.post("/login", async (req, res) => {
    try {
        const { email, password } = req.body;

        const user = await User.findOne({ email });
        if (!user)
            return res.status(400).json({ message: "Invalid credentials" });

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch)
            return res.status(400).json({ message: "Invalid credentials" });

        const token = jwt.sign(
            { id: user._id, role: user.role },
            process.env.JWT_SECRET,
            { expiresIn: "1d" }
        );

        res.json({
            token,
            role: user.role,
            name: user.name,
            email: user.email
        });

    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});


// ================= PATIENT DASHBOARD =================
router.get(
    "/patient",
    authMiddleware,
    roleMiddleware("patient"),
    async(req, res) => {
        const user = await User.findById(req.user.id).select("-password");
        res.json({
            message: "Welcome Patient Dashboard",
            role: user.role,
            name: user.name,
            email: user.email
        });
    }
);


// ================= DOCTOR DASHBOARD =================
router.get(
    "/doctor",
    authMiddleware,
    roleMiddleware("doctor"),
    async(req, res) => {
        const user = await User.findById(req.user.id).select("-password");
        res.json({
            message: "Welcome Doctor Dashboard",
            role: user.role,
            name: user.name,
            email: user.email
        });
    }
);

module.exports = router;
