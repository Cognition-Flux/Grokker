"use client";
import React from 'react';
import './Loader.css';

export type LoaderProps = {
}

const Loader: React.FC<LoaderProps> = ({ }) => {
	return (
		<div id="page">
			<div id="container">
				<div id="ring"></div>
				<div id="ring"></div>
				<div id="ring"></div>
				<div id="ring"></div>
				<div id="h3">t-Agentic</div>
			</div>
		</div>
	);
};

export default Loader;
