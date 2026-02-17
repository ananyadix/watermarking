const mongoose = require("mongoose");

const doctorSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true
    },

    email: {
        type: String,
        required: true,
        unique: true
    },

    password: {
        type: String,
        required: true
    },

    phone: {
        type: String,
        required: true
    },

    specialization: {
        type: String,
        required: true
    },

    qualification: {
        type: String,
        required: true
    },

    licenseNumber: {
        type: String,
        required: true,
        unique: true
    },

    experience: {
        type: Number,   // in years
        required: true
    }

}, { timestamps: true });

module.exports = mongoose.model("Doctor", doctorSchema);
